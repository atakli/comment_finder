"""Çalıştırma geçmişi: çekilen ham veri ve ayıklama sonuçları data/runs/<id>/ altında JSON.

Her çalıştırma bir dizin:
- meta.json — liste için küçük özet (kaynak, linkler, prompt, model, sayılar)
- data.json — {"fetched": {link: [öğe]}, "results": [öğe] | null, "errors": [str]}
Silinen kayıtlar data/trash/<id>/ altına taşınır, restore_run ile geri alınabilir.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "data" / "runs"
TRASH_DIR = Path(__file__).parent / "data" / "trash"


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
             fetched: dict, results: list | None, errors: list[str],
             thinking_level: str | None = None) -> None:
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
    if thinking_level:
        meta["thinking_level"] = thinking_level
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
    """Kaydı çöp kutusuna taşır (kalıcı silmez)."""
    src, dst = RUNS_DIR / run_id, TRASH_DIR / run_id
    if not src.exists():
        return
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dst, ignore_errors=True)  # aynı id daha önce silinip geri yüklendiyse eski kopya
    src.replace(dst)


def restore_run(run_id: str) -> bool:
    """Çöp kutusundaki kaydı geri taşır; kayıt yoksa ya da aynı id zaten varsa False."""
    src, dst = TRASH_DIR / run_id, RUNS_DIR / run_id
    if not src.exists() or dst.exists():
        return False
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    src.replace(dst)
    return True


def merge_fetched(base_fetched: dict, new_fetched: dict) -> tuple[dict, int]:
    """İki fetched dict'ini birleştirir ve öğeleri (id veya yazar+metin) tekilleştirir.

    Döner:
        (birlesmis_fetched, yeni_eklenen_oge_sayisi)
    """
    merged = {k: list(v) for k, v in (base_fetched or {}).items()}
    total_added = 0
    for link, new_items in (new_fetched or {}).items():
        if link in merged:
            existing_keys = {
                (it.get("source"), str(it.get("id"))) if it.get("id") is not None
                else (it.get("source"), str(it.get("author")), str(it.get("text")))
                for it in merged[link]
            }
            items_to_add = [
                it for it in new_items
                if ((it.get("source"), str(it.get("id"))) if it.get("id") is not None
                    else (it.get("source"), str(it.get("author")), str(it.get("text")))) not in existing_keys
            ]
            merged[link] = merged[link] + items_to_add
            total_added += len(items_to_add)
        else:
            merged[link] = list(new_items)
            total_added += len(new_items)
    return merged, total_added


def merge_results(base_results: list | None, new_results: list | None) -> tuple[list, int]:
    """İki süzülmüş sonuç listesini birleştirir, tekilleştirir ve puana göre sıralar.

    Döner:
        (birlesmis_results, yeni_eklenen_sonuc_sayisi)
    """
    base = list(base_results or [])
    new = list(new_results or [])
    if not base:
        sorted_new = sorted(new, key=lambda r: (-r.get("score", 0), -r.get("likes", 0)))
        return sorted_new, len(new)
    if not new:
        return base, 0

    existing_keys = {
        (r.get("source"), str(r.get("id"))) if r.get("id") is not None
        else (r.get("source"), str(r.get("author")), str(r.get("text")))
        for r in base
    }
    items_to_add = [
        r for r in new
        if ((r.get("source"), str(r.get("id"))) if r.get("id") is not None
            else (r.get("source"), str(r.get("author")), str(r.get("text")))) not in existing_keys
    ]
    combined = base + items_to_add
    sorted_combined = sorted(combined, key=lambda r: (-r.get("score", 0), -r.get("likes", 0)))
    return sorted_combined, len(items_to_add)

