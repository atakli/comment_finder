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
from pydantic import BaseModel

MODELS = {
    "Claude Opus 5": "claude-opus-5",
    "Claude Sonnet 5": "claude-sonnet-5",
    "Claude Haiku 4.5": "claude-haiku-4-5",
}

SYSTEM = """Sen bir içerik ayıklama asistanısın. Kullanıcı sana bir YouTube videosunun yorumlarını \
veya bir Ekşi Sözlük başlığının entry'lerini ve bir ayıklama kriteri verecek.
Her öğe [numara] ile başlar. Kritere gerçekten uyan, işe yarar öğeleri seç; uymayanları dahil etme.
Her seçilen öğe için: numarası, 1-10 arası uygunluk puanı ve kısa (tek cümle, Türkçe) seçilme gerekçesi ver.
Hiçbiri uymuyorsa boş liste döndür."""


class Pick(BaseModel):
    index: int
    score: int
    reason: str


class Picks(BaseModel):
    selected: list[Pick]


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


def _call_api(model: str, message: str) -> Picks:
    client = anthropic.Anthropic()
    kwargs = {}
    if model == "claude-opus-5":
        # Opus 5 bir isteği reddederse sunucu tarafında otomatik olarak başka modele düşer
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": message}],
        output_format=Picks,
        **kwargs,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Model isteği reddetti.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Yanıt max_tokens sınırına takıldı; parti boyutunu küçültün.")
    return response.parsed_output


def _call_cli(model: str, message: str) -> Picks:
    cmd = [
        "claude", "-p", SYSTEM,
        "--output-format", "json",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--json-schema", json.dumps(Picks.model_json_schema()),
    ]
    # Proje CLAUDE.md'si bağlama karışmasın diye boş bir dizinde çalıştır
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=message, capture_output=True, text=True, cwd=cwd, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI hatası: {proc.stderr.strip() or proc.stdout.strip()[:500]}")
    data = json.loads(proc.stdout)
    if data.get("is_error") or "structured_output" not in data:
        raise RuntimeError(f"claude CLI yapılandırılmış çıktı döndürmedi: {str(data.get('result'))[:500]}")
    return Picks.model_validate(data["structured_output"])


def filter_items(items: list[dict], criteria: str, model: str, backend: str,
                 batch_size: int = 150, workers: int = 4, on_progress=None) -> tuple[list[dict], list[str]]:
    """(seçilenler, hatalar) döndürür. Seçilenlere score/reason eklenir, puana göre sıralanır."""
    call = _call_api if backend == "api" else _call_cli
    batches = [(o, items[o:o + batch_size]) for o in range(0, len(items), batch_size)]
    results, errors, done = [], [], 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, model, _user_message(criteria, b, o)): o for o, b in batches}
        for fut in as_completed(futures):
            try:
                for p in fut.result().selected:
                    if 0 <= p.index < len(items):
                        results.append({**items[p.index], "score": p.score, "reason": p.reason})
            except Exception as e:  # bir partinin hatası diğerlerini durdurmasın
                errors.append(f"Parti {futures[fut]}: {e}")
            done += 1
            if on_progress:
                on_progress(done, len(batches))

    # Aynı öğe birden fazla kez seçildiyse tekilleştir
    unique = {(r["source"], r["id"]): r for r in results}
    ranked = sorted(unique.values(), key=lambda r: (-r["score"], -r["likes"]))
    return ranked, errors
