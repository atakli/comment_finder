"""Yorumları kullanıcının prompt'una göre Claude ile ayıklar.

İki backend:
- "api": Anthropic SDK (ANTHROPIC_API_KEY gerekir)
- "cli": yerel `claude -p` (Claude Code aboneliğiyle çalışır, API anahtarı gerekmez)
"""
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from pydantic import BaseModel, Field

MODELS = {
    "Claude Opus 5": "claude-opus-5",
    "Claude Sonnet 5": "claude-sonnet-5",
    "Claude Haiku 4.5": "claude-haiku-4-5",
}

# (giriş, çıkış) $ / 1M token
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# Maliyet tahmini varsayımları
CHARS_PER_TOKEN = 3.0          # Türkçe metinde kabaca; tam sayım için count_tokens kullanılır
SCHEMA_OVERHEAD = 250          # yapılandırılmış çıktı şemasının sisteme eklediği token
PICK_TOKENS = 45               # seçilen öğe başına çıktı (index, puan, tek cümlelik gerekçe)
SELECT_RATIO = (0.1, 0.5)      # seçilme oranı aralığı
THINKING_TOKENS = (300, 3000)  # parti başına düşünme tokenı (Haiku 4.5'te düşünme kapalı)

# Her prompt'un sonuna sabit eklenir (arayüzde görünür, kullanıcı tekrar yazmasın)
TRANSLATE_SUFFIX = "seçtiğin yorumlardan türkçe olmayanları türkçeye çevir"

SYSTEM = """Sen bir içerik ayıklama asistanısın. Kullanıcı sana bir YouTube videosunun yorumlarını \
veya bir Ekşi Sözlük başlığının entry'lerini ve bir ayıklama kriteri verecek.
Her öğe [numara] ile başlar. Kritere gerçekten uyan, işe yarar öğeleri seç; uymayanları dahil etme.
Her seçilen öğe için: numarası, 1-10 arası uygunluk puanı ve kısa (tek cümle, Türkçe) seçilme gerekçesi ver.
Kriter çeviri istiyorsa, Türkçe olmayan seçilmiş öğelerin tam Türkçe çevirisini translation alanına yaz; \
Türkçe olanlar için translation boş string olsun.
Hiçbiri uymuyorsa boş liste döndür."""


class Pick(BaseModel):
    index: int
    score: int
    reason: str
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


def count_input_tokens(model: str, messages: list[str], workers: int = 4) -> int:
    """Anthropic token sayma API'si ile tam giriş tokenı (ücretsiz, API kimlik bilgisi gerekir)."""
    client = anthropic.Anthropic()

    def count(msg):
        return client.messages.count_tokens(
            model=model, system=SYSTEM, messages=[{"role": "user", "content": msg}]).input_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(count, messages)) + SCHEMA_OVERHEAD * len(messages)


def estimate_cost(model: str, messages: list[str], n_items: int, input_tokens: int | None = None) -> dict:
    """Ayıklamadan önce kabaca maliyet tahmini. input_tokens verilmezse karakterden hesaplanır."""
    n_batches = len(messages)
    chars = sum(len(m) for m in messages) + len(SYSTEM) * n_batches
    exact = input_tokens is not None
    if not exact:
        input_tokens = round(chars / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD * n_batches
    thinking = (0, 0) if model == "claude-haiku-4-5" else THINKING_TOKENS
    out_lo = round(n_items * SELECT_RATIO[0] * PICK_TOKENS) + thinking[0] * n_batches
    out_hi = round(n_items * SELECT_RATIO[1] * PICK_TOKENS) + thinking[1] * n_batches
    price_in, price_out = PRICES[model]
    cost = lambda out: (input_tokens * price_in + out * price_out) / 1_000_000
    return {"batches": n_batches, "chars": chars, "input_tokens": input_tokens, "exact": exact,
            "output_tokens": (out_lo, out_hi), "cost": (cost(out_lo), cost(out_hi))}


def _call_api(model: str, message: str, system: str = SYSTEM, schema: type[BaseModel] = Picks):
    client = anthropic.Anthropic()
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


def _call_cli(model: str, message: str, system: str = SYSTEM, schema: type[BaseModel] = Picks):
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


def filter_items(items: list[dict], criteria: str, model: str, backend: str,
                 batch_size: int = 150, workers: int = 4, on_progress=None) -> tuple[list[dict], list[str]]:
    """(seçilenler, hatalar) döndürür. Seçilenlere score/reason eklenir, puana göre sıralanır."""
    call = _call_api if backend == "api" else _call_cli
    batches = build_requests(items, criteria, batch_size)
    results, errors, done = [], [], 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, model, msg): o for o, msg in batches}
        for fut in as_completed(futures):
            try:
                for p in fut.result().selected:
                    if 0 <= p.index < len(items):
                        picked = {**items[p.index], "score": p.score, "reason": p.reason}
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
                    on_progress=None) -> tuple[list[dict], list[str]]:
    """Önceden seçilmiş öğelerden Türkçe olmayanların metnini Türkçe çevirisiyle değiştirir.
    (yeni öğe listesi, hatalar) döndürür; hatalı partideki öğeler kontrol edilmemiş kalır."""
    call = _call_api if backend == "api" else _call_cli
    out = [dict(it) for it in items]
    batches = [(o, "<yorumlar>\n" + _format_batch(items[o:o + batch_size], o) + "\n</yorumlar>")
               for o in range(0, len(items), batch_size)]
    errors, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, model, msg, TRANSLATE_SYSTEM, Translations): o for o, msg in batches}
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
