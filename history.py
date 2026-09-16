"""Çalıştırma geçmişi: çekilen ham veri ve ayıklama sonuçları data/runs/<id>/ altında JSON.

Her çalıştırma bir dizin:
- meta.json — liste için küçük özet (kaynak, linkler, prompt, model, sayılar)
- data.json — {"fetched": {link: [öğe]}, "results": [öğe] | null, "errors": [str]}
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "data" / "runs"


def new_run_id() -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id, n = base, 1
    while (RUNS_DIR / run_id).exists():
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def save_run(run_id: str, *, source: str, criteria: str, model: str, backend: str,
             fetched: dict, results: list | None, errors: list[str]) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": run_id,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "links": list(fetched),
        "titles": [items[0]["title"] for items in fetched.values() if items],
        "criteria": criteria,
        "model": model,
        "backend": backend,
        "n_items": sum(len(items) for items in fetched.values()),
        "n_selected": None if results is None else len(results),
    }
    _write_json(run_dir / "data.json", {"fetched": fetched, "results": results, "errors": errors})
    _write_json(run_dir / "meta.json", meta)


def list_runs() -> list[dict]:
    """Yeniden eskiye meta listesi."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for meta_path in sorted(RUNS_DIR.glob("*/meta.json"), reverse=True):
        try:
            runs.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def load_run(run_id: str) -> tuple[dict, dict]:
    run_dir = RUNS_DIR / run_id
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    data = json.loads((run_dir / "data.json").read_text(encoding="utf-8"))
    return meta, data


def delete_run(run_id: str) -> None:
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)
