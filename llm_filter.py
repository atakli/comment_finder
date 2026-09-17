"""Yorumları kullanıcının prompt'una göre LLM ile ayıklar.

Sağlayıcı başına backend:
- anthropic: "api" (Anthropic SDK) ya da "cli" (yerel `claude -p`, Claude Code aboneliğiyle çalışır)
- google: yalnızca "api" (google-genai SDK)
API anahtarı ortam değişkeninde yoksa arayüz kullanıcıdan ister ve çağrılara doğrudan iletilir
(diske/geçmişe yazılmaz).
"""
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from pydantic import BaseModel, Field

# Sağlayıcı başına API anahtarı ortam değişkeni + arayüzde gösterilecek ad
PROVIDERS = {
    "anthropic": {"env": "ANTHROPIC_API_KEY", "label": "Anthropic"},
    "google": {"env": "GEMINI_API_KEY", "label": "Google Gemini"},
}

MODELS = {
    "Claude Opus 5": {"id": "claude-opus-5", "provider": "anthropic"},
    "Claude Sonnet 5": {"id": "claude-sonnet-5", "provider": "anthropic"},
    "Claude Haiku 4.5": {"id": "claude-haiku-4-5", "provider": "anthropic"},
    "Gemini 3.8 Flash": {"id": "gemini-3.8-flash", "provider": "google"},
    "Gemini 3.1 Flash-Lite (Preview)": {"id": "models/gemini-3.1-flash-lite-preview", "provider": "google"},
}
MODEL_PROVIDER = {v["id"]: v["provider"] for v in MODELS.values()}
MODEL_PROVIDER.update({
    "gemini-3.1-flash-lite-preview": "google",
    "gemini-3.1-flash-preview": "google",
    "gemini-3.8-flash": "google",
    "models/gemini-3.8-flash": "google",
})

# (giriş, çıkış) $ / 1M token
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gemini-3.8-flash": (0.75, 3.75),
    "models/gemini-3.8-flash": (0.75, 3.75),
    "models/gemini-3.1-flash-lite-preview": (0.075, 0.30),  # tahmini; önizleme modeli, resmi fiyatı teyit edilmedi
    "gemini-3.1-flash-lite-preview": (0.075, 0.30),
    "gemini-3.1-flash-preview": (0.075, 0.30),
}
# Maliyet tahmini varsayımları
CHARS_PER_TOKEN = 3.0          # Türkçe metinde kabaca; tam sayım için count_tokens kullanılır
SCHEMA_OVERHEAD = 250          # yapılandırılmış çıktı şemasının sisteme eklediği token
BASE_PICK_TOKENS = 12          # seçilen öğe başına çıktı: indeks + puan
GROUP_TOKENS = 10              # seçilen öğe başına ek çıktı: kısa grup adı (gruplama özelliğinin maliyeti)
PICK_TOKENS = BASE_PICK_TOKENS + GROUP_TOKENS
SELECT_RATIO = (0.1, 0.5)      # seçilme oranı aralığı
THINKING_TOKENS = (300, 3000)  # parti başına düşünme tokenı (Haiku 4.5'ta düşünme kapalı)
NO_THINKING_MODELS = {
    "claude-haiku-4-5",
}

# Modellerin düşünme (thinking / reasoning) parametresi yapılandırması
MODEL_THINKING_CONFIG = {
    "gemini-3.8-flash": {
        "type": "level",
        "mandatory": True,  # Düşünme modu zorunlu (kapatılamaz), ancak seviyesi seçilebilir
        "levels": ["LOW", "MEDIUM", "HIGH"],
        "default": "MEDIUM",
        "labels": {
            "LOW": "Düşük (Hızlı, daha az token)",
            "MEDIUM": "Orta (Dengeli - Varsayılan)",
            "HIGH": "Yüksek (Derin akıl yürütme)",
        },
        "token_ranges": {
            "LOW": (150, 1000),
            "MEDIUM": (300, 3000),
            "HIGH": (1000, 6000),
        },
    },
    "models/gemini-3.8-flash": {
        "type": "level",
        "mandatory": True,
        "levels": ["LOW", "MEDIUM", "HIGH"],
        "default": "MEDIUM",
        "labels": {
            "LOW": "Düşük (Hızlı, daha az token)",
            "MEDIUM": "Orta (Dengeli - Varsayılan)",
            "HIGH": "Yüksek (Derin akıl yürütme)",
        },
        "token_ranges": {
            "LOW": (150, 1000),
            "MEDIUM": (300, 3000),
            "HIGH": (1000, 6000),
        },
    },
    "models/gemini-3.1-flash-lite-preview": {
        "type": "level",
        "mandatory": False,
        "levels": ["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        "default": "MINIMAL",
        "labels": {
            "MINIMAL": "Minimal (En hızlı, varsayılan)",
            "LOW": "Düşük",
            "MEDIUM": "Orta",
            "HIGH": "Yüksek (Derin akıl yürütme)",
        },
        "token_ranges": {
            "MINIMAL": (0, 200),
            "LOW": (150, 1000),
            "MEDIUM": (300, 3000),
            "HIGH": (1000, 6000),
        },
    },
    "gemini-3.1-flash-lite-preview": {
        "type": "level",
        "mandatory": False,
        "levels": ["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        "default": "MINIMAL",
        "labels": {
            "MINIMAL": "Minimal (En hızlı, varsayılan)",
            "LOW": "Düşük",
            "MEDIUM": "Orta",
            "HIGH": "Yüksek (Derin akıl yürütme)",
        },
        "token_ranges": {
            "MINIMAL": (0, 200),
            "LOW": (150, 1000),
            "MEDIUM": (300, 3000),
            "HIGH": (1000, 6000),
        },
    },
}


def get_model_thinking_config(model: str) -> dict | None:
    """Model düşünme parametresi kabul ediyorsa yapılandırmasını döner."""
    return MODEL_THINKING_CONFIG.get(model)

# Her prompt'un sonuna sabit eklenir (arayüzde görünür, kullanıcı tekrar yazmasın)
TRANSLATE_SUFFIX = "seçtiğin yorumlardan türkçe olmayanları türkçeye çevir"

SYSTEM = """Sen bir içerik ayıklama asistanısın. Kullanıcı sana bir YouTube videosunun yorumlarını \
veya bir Ekşi Sözlük başlığının entry'lerini ve bir ayıklama kriteri verecek.
Her öğe [numara] ile başlar. Kritere gerçekten uyan, işe yarar öğeleri seç; uymayanları dahil etme.
Seçtiğin her öğeyi ortak temasına göre kısa bir Türkçe grup adına ata (ör. "hacamat önerenler", \
"doktora gitmeyi tavsiye edenler"). Az sayıda, genel grup adı kullan; birbirine benzer öğelerde grup adını \
harfiyen aynı yaz ki aynı grupta toplansınlar. Her öğeye ayrı bir grup uydurma.
Her seçilen öğe için: numarası, 1-10 arası uygunluk puanı ve grup adı ver.
Kriter çeviri istiyorsa, Türkçe olmayan seçilmiş öğelerin tam Türkçe çevirisini translation alanına yaz; \
Türkçe olanlar için translation boş string olsun.
Hiçbiri uymuyorsa boş liste döndür."""


class Pick(BaseModel):
    index: int
    score: int
    group: str = Field(description="Öğenin ait olduğu kısa Türkçe grup/tema adı; aynı temadaki öğelerde "
                                    "birebir aynı ifadeyi kullan")
    translation: str = Field(description="Öğe Türkçe değilse tam Türkçe çevirisi, Türkçeyse boş string")


class Picks(BaseModel):
    selected: list[Pick]


TRANSLATE_SYSTEM = """Sana [numara] ile başlayan yorumlar verilecek. Türkçe olmayan her yorumu (Hintçe, \
Latin harfli Hintçe/Urduca, İngilizce vb.) anlamını koruyarak doğal Türkçeye çevir.
Türkçe olan yorumları listeye ekleme. Hiçbiri çevrilecek değilse boş liste döndür."""


class Translation(BaseModel):
    index: int
    translation: str


class Translations(BaseModel):
    translated: list[Translation]


def strip_translate_suffix(criteria: str) -> str:
    """Prompt'un sonundaki sabit çeviri ekini (varsa) çıkarır."""
    text = (criteria or "").rstrip()
    if text.lower().endswith(TRANSLATE_SUFFIX):
        text = text[:-len(TRANSLATE_SUFFIX)].rstrip()
    return text


def with_translate_suffix(criteria: str) -> str:
    """Kullanıcı prompt'u + sabit çeviri eki (ek zaten yazılmışsa ikinci kez eklenmez)."""
    text = strip_translate_suffix(criteria)
    return f"{text}\n\n{TRANSLATE_SUFFIX}" if text else TRANSLATE_SUFFIX


def default_backend() -> str:
    return "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"


def _format_batch(items: list[dict], offset: int) -> str:
    lines = []
    for i, it in enumerate(items):
        meta = f"(beğeni/fav: {it['likes']}{', yanıt' if it['is_reply'] else ''})"
        lines.append(f"[{offset + i}] {meta} {it['text']}")
    return "\n\n".join(lines)


def _user_message(criteria: str, items: list[dict], offset: int) -> str:
    kind = "Ekşi Sözlük entry'leri" if items[0]["source"] == "eksi" else "YouTube yorumları"
    return (
        f"Kaynak: {items[0]['title']} ({kind})\n\n"
        f"<kriter>\n{criteria}\n</kriter>\n\n"
        f"<ogeler>\n{_format_batch(items, offset)}\n</ogeler>"
    )


def build_requests(items: list[dict], criteria: str, batch_size: int) -> list[tuple[int, str]]:
    """Öğeleri partilere böler; (offset, kullanıcı mesajı) listesi döndürür."""
    return [(o, _user_message(criteria, items[o:o + batch_size], o)) for o in range(0, len(items), batch_size)]


def count_input_tokens(model: str, messages: list[str], workers: int = 4, api_key: str | None = None) -> int:
    """Anthropic token sayma API'si ile tam giriş tokenı (ücretsiz, API kimlik bilgisi gerekir)."""
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def count(msg):
        return client.messages.count_tokens(
            model=model, system=SYSTEM, messages=[{"role": "user", "content": msg}]).input_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(count, messages)) + SCHEMA_OVERHEAD * len(messages)


def estimate_cost(model: str, messages: list[str], n_items: int, input_tokens: int | None = None,
                  thinking_level: str | None = None) -> dict:
    """Ayıklamadan önce kabaca maliyet tahmini. input_tokens verilmezse karakterden hesaplanır."""
    n_batches = len(messages)
    chars = sum(len(m) for m in messages) + len(SYSTEM) * n_batches
    exact = input_tokens is not None
    if not exact:
        input_tokens = round(chars / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD * n_batches
    th_cfg = get_model_thinking_config(model)
    if th_cfg:
        lvl = (thinking_level or th_cfg["default"]).upper()
        token_ranges = th_cfg.get("token_ranges", {})
        thinking = token_ranges.get(lvl, THINKING_TOKENS)
    elif model in NO_THINKING_MODELS:
        thinking = (0, 0)
    else:
        thinking = THINKING_TOKENS
    out_lo = round(n_items * SELECT_RATIO[0] * PICK_TOKENS) + thinking[0] * n_batches
    out_hi = round(n_items * SELECT_RATIO[1] * PICK_TOKENS) + thinking[1] * n_batches
    price_in, price_out = PRICES[model]
    cost = lambda out: (input_tokens * price_in + out * price_out) / 1_000_000
    # Gruplama özelliğinin (GROUP_TOKENS) çıktı maliyetine eklediği pay, ayrı gösterebilmek için
    group_lo = round(n_items * SELECT_RATIO[0] * GROUP_TOKENS)
    group_hi = round(n_items * SELECT_RATIO[1] * GROUP_TOKENS)
    group_cost = (cost(out_lo) - cost(out_lo - group_lo), cost(out_hi) - cost(out_hi - group_hi))
    return {"batches": n_batches, "chars": chars, "input_tokens": input_tokens, "exact": exact,
            "output_tokens": (out_lo, out_hi), "cost": (cost(out_lo), cost(out_hi)), "group_cost": group_cost}


def _call_api(model: str, message: str, system: str = SYSTEM, schema: type[BaseModel] = Picks,
              api_key: str | None = None, thinking_level: str | None = None):
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    kwargs = {}
    if model == "claude-opus-5":
        # Opus 5 bir isteği reddederse sunucu tarafında otomatik olarak başka modele düşer
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": message}],
        output_format=schema,
        **kwargs,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Model isteği reddetti.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Yanıt max_tokens sınırına takıldı; parti boyutunu küçültün.")
    return response.parsed_output


def _call_cli(model: str, message: str, system: str = SYSTEM, schema: type[BaseModel] = Picks,
              api_key: str | None = None, thinking_level: str | None = None):
    cmd = [
        "claude", "-p", system,
        "--output-format", "json",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--json-schema", json.dumps(schema.model_json_schema()),
    ]
    # Proje CLAUDE.md'si bağlama karışmasın diye boş bir dizinde çalıştır
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=message, capture_output=True, text=True, cwd=cwd, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI hatası: {proc.stderr.strip() or proc.stdout.strip()[:500]}")
    data = json.loads(proc.stdout)
    if data.get("is_error") or "structured_output" not in data:
        raise RuntimeError(f"claude CLI yapılandırılmış çıktı döndürmedi: {str(data.get('result'))[:500]}")
    return schema.model_validate(data["structured_output"])


def _call_gemini(model: str, message: str, system: str = SYSTEM, schema: type[BaseModel] = Picks,
                 api_key: str | None = None, thinking_level: str | None = None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    config_kwargs = {
        "system_instruction": system,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }

    # Model düşünme parametresi kabul ediyorsa thinking_config ekle
    th_cfg = get_model_thinking_config(model)
    eff_level = thinking_level or (th_cfg["default"] if th_cfg else None)
    if eff_level:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=eff_level.upper())
        except Exception:
            pass

    response = client.models.generate_content(
        model=model,
        contents=message,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    if response.parsed is None:
        raise RuntimeError(f"Gemini yapılandırılmış çıktı döndürmedi: {str(response.text)[:500]}")
    return response.parsed


def _dispatch(model: str, backend: str):
    if MODEL_PROVIDER.get(model) == "google":
        return _call_gemini
    return _call_api if backend == "api" else _call_cli


def filter_items(items: list[dict], criteria: str, model: str, backend: str, batch_size: int = 150,
                 workers: int = 4, on_progress=None, api_key: str | None = None,
                 thinking_level: str | None = None) -> tuple[list[dict], list[str]]:
    """(seçilenler, hatalar) döndürür. Seçilenlere score/group eklenir, puana göre sıralanır."""
    call = _dispatch(model, backend)
    batches = build_requests(items, criteria, batch_size)
    results, errors, done = [], [], 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, model, msg, SYSTEM, Picks, api_key, thinking_level): o for o, msg in batches}
        for fut in as_completed(futures):
            try:
                for p in fut.result().selected:
                    if 0 <= p.index < len(items):
                        picked = {**items[p.index], "score": p.score, "group": p.group}
                        if p.translation.strip():  # Türkçe olmayan: yalnızca çevirisi tutulur
                            picked["text"] = p.translation.strip()
                        picked["translated"] = bool(p.translation.strip())
                        results.append(picked)
            except Exception as e:  # bir partinin hatası diğerlerini durdurmasın
                errors.append(f"Parti {futures[fut]}: {e}")
            done += 1
            if on_progress:
                on_progress(done, len(batches))

    # Aynı öğe birden fazla kez seçildiyse tekilleştir
    unique = {(r["source"], r["id"]): r for r in results}
    ranked = sorted(unique.values(), key=lambda r: (-r["score"], -r["likes"]))
    return ranked, errors


def needs_translation_check(results: list[dict] | None) -> bool:
    """Çeviri özelliğinden önce ayıklanmış (dil kontrolü yapılmamış) sonuçlar var mı."""
    return bool(results) and any("translated" not in r for r in results)


def translate_items(items: list[dict], model: str, backend: str, batch_size: int = 40, workers: int = 4,
                    on_progress=None, api_key: str | None = None,
                    thinking_level: str | None = None) -> tuple[list[dict], list[str]]:
    """Önceden seçilmiş öğelerden Türkçe olmayanların metnini Türkçe çevirisiyle değiştirir.
    (yeni öğe listesi, hatalar) döndürür; hatalı partideki öğeler kontrol edilmemiş kalır."""
    call = _dispatch(model, backend)
    out = [dict(it) for it in items]
    batches = [(o, "<yorumlar>\n" + _format_batch(items[o:o + batch_size], o) + "\n</yorumlar>")
               for o in range(0, len(items), batch_size)]
    errors, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, model, msg, TRANSLATE_SYSTEM, Translations, api_key, thinking_level): o for o, msg in batches}
        for fut in as_completed(futures):
            o = futures[fut]
            try:
                translations = {t.index: t.translation.strip() for t in fut.result().translated}
                for i in range(o, min(o + batch_size, len(items))):
                    if "translated" in out[i]:
                        continue
                    if translations.get(i):
                        out[i]["text"] = translations[i]
                    out[i]["translated"] = bool(translations.get(i))
            except Exception as e:
                errors.append(f"Çeviri partisi {o}: {e}")
            done += 1
            if on_progress:
                on_progress(done, len(batches))
    return out, errors
