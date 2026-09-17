"""Kalıcı dosya depolaması ile Streamlit API anahtarları yönetimi.

Birden fazla model ve sağlayıcı için isimlendirilmiş API anahtarlarının bilgisayarda
kalıcı bir dosyada (varsayılan: data/api_keys.json) saklanmasını ve model bazlı varsayılan
anahtar seçimini sağlar. Dosya git'e eklenmez (.gitignore'dadır).
"""
import json
import os
from pathlib import Path
import uuid
import streamlit as st
from llm_filter import MODELS, PROVIDERS


def get_api_keys_file() -> Path:
    """API anahtarlarının kaydedileceği kalıcı dosya yolunu döner."""
    custom = os.environ.get("API_KEYS_FILE")
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent / "data" / "api_keys.json"


def sanitize_model_key(model_id: str) -> str:
    """Widget anahtarları için model kimliğini güvenli dizeye dönüştürür."""
    return model_id.replace("/", "_").replace(" ", "_").replace(".", "_").replace("-", "_")


def mask_key(key: str | None) -> str:
    """API anahtarını arayüzde güvenli göstermek için maskeler."""
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "••••"
    return f"{key[:6]}...{key[-4:]}"


def _migrate_old_storage(old_data: dict) -> dict:
    """Eski tekil/sözlük formatındaki kayıtlı anahtarları sürüm 2 formatına taşır."""
    new_data = {
        "version": 2,
        "keys": [],
        "default_keys": {},
    }
    for k, v in old_data.items():
        if not v or not isinstance(v, str):
            continue
        val = v.strip()
        if not val:
            continue

        # Sağlayıcı ve model tespiti
        target_model_id = None
        if k in MODELS:
            provider = MODELS[k]["provider"]
            target_model_id = MODELS[k]["id"]
            name = f"{k} Anahtarı"
        elif k in PROVIDERS:
            provider = k
            name = f"{PROVIDERS[k]['label']} Anahtarı"
        elif val.startswith("sk-ant"):
            provider = "anthropic"
            name = "Anthropic Anahtarı"
        elif val.startswith("AIza"):
            provider = "google"
            name = "Google Gemini Anahtarı"
        else:
            provider = "anthropic"
            name = f"Kayıtlı Anahtar ({k})"

        # Aynı anahtar zaten eklendi mi?
        existing = next((item for item in new_data["keys"] if item["key"] == val), None)
        if existing:
            key_id = existing["id"]
        else:
            key_id = f"key_{uuid.uuid4().hex[:8]}"
            new_data["keys"].append({
                "id": key_id,
                "name": name,
                "provider": provider,
                "key": val,
            })

        if target_model_id:
            new_data["default_keys"][target_model_id] = key_id
        elif provider:
            for _, info in MODELS.items():
                mid = info["id"]
                if info["provider"] == provider and mid not in new_data["default_keys"]:
                    new_data["default_keys"][mid] = key_id

    return new_data


def load_keys_from_disk() -> dict:
    """Diskteki kalıcı dosyadan API anahtarlarını yükler."""
    keys_file = get_api_keys_file()
    if not keys_file.exists():
        # Alternatif olarak kök dizindeki .api_keys.json kontrolü
        alt_file = Path(__file__).resolve().parent / ".api_keys.json"
        if alt_file.exists():
            try:
                data = json.loads(alt_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data if "keys" in data else _migrate_old_storage(data)
            except Exception:
                pass
        return {"version": 2, "keys": [], "default_keys": {}}

    try:
        data = json.loads(keys_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 2, "keys": [], "default_keys": {}}
        if "keys" not in data:
            data = _migrate_old_storage(data)
        return data
    except Exception:
        return {"version": 2, "keys": [], "default_keys": {}}


def save_keys_to_disk(data: dict) -> None:
    """API anahtarlarını bilgisayardaki kalıcı dosyaya güvenli (atomik) şekilde yazar."""
    keys_file = get_api_keys_file()
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = keys_file.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp_path.chmod(0o600)  # Yalnızca dosya sahibi okuyup yazabilsin
    except OSError:
        pass
    tmp_path.replace(keys_file)


def get_api_keys_data() -> dict:
    """Oturumdaki API anahtarı verisini döner; başlatılmamışsa diskten yükler."""
    state = getattr(st, "session_state", None)
    if state is None:
        return load_keys_from_disk()
    if "api_keys_data" not in state:
        state.api_keys_data = load_keys_from_disk()
    return state.api_keys_data


def get_all_keys(provider: str | None = None) -> list[dict]:
    """Kayıtlı tüm API anahtarlarını döner. İsteğe göre sağlayıcıya göre süzer."""
    data = get_api_keys_data()
    keys = data.get("keys", [])
    if provider:
        return [k for k in keys if k.get("provider") == provider]
    return list(keys)


def get_key_by_id(key_id: str) -> dict | None:
    """Kimliğe göre kayıtlı anahtarı döner."""
    data = get_api_keys_data()
    for k in data.get("keys", []):
        if k.get("id") == key_id:
            return k
    return None


def get_default_key_id_for_model(model_id: str, provider: str | None = None) -> str | None:
    """Verilen model için varsayılan anahtar kimliğini döner."""
    data = get_api_keys_data()
    default_keys = data.get("default_keys", {})
    def_id = default_keys.get(model_id)
    if def_id:
        if get_key_by_id(def_id):
            return def_id

    # Model için atanmamışsa sağlayıcının ilk anahtarını dönebilir
    if provider:
        prov_keys = get_all_keys(provider)
        if prov_keys:
            return prov_keys[0]["id"]

    return None


def set_default_key_for_model(model_id: str, key_id: str | None):
    """Verilen model için varsayılan anahtarı ayarlar veya kaldırır ve kalıcı dosyaya yazar."""
    data = get_api_keys_data()
    data.setdefault("default_keys", {})
    if key_id and key_id != "none":
        data["default_keys"][model_id] = key_id
    else:
        data["default_keys"].pop(model_id, None)

    save_keys_to_disk(data)
    if hasattr(st, "session_state"):
        st.session_state.api_keys_data = data


def add_api_key(name: str, key: str, provider: str, default_for_models: list[str] | None = None) -> str:
    """Yeni bir API anahtarı kaydeder ve kalıcı dosyaya yazar."""
    data = get_api_keys_data()
    key_id = f"key_{uuid.uuid4().hex[:8]}"
    clean_name = name.strip() or f"{PROVIDERS.get(provider, {}).get('label', provider)} Anahtarı"
    new_entry = {
        "id": key_id,
        "name": clean_name,
        "provider": provider,
        "key": key.strip(),
    }
    data.setdefault("keys", []).append(new_entry)

    if default_for_models:
        data.setdefault("default_keys", {})
        for m in default_for_models:
            data["default_keys"][m] = key_id
            safe_m = sanitize_model_key(m)
            if hasattr(st, "session_state"):
                st.session_state[f"active_key_select_{safe_m}"] = key_id

    save_keys_to_disk(data)
    if hasattr(st, "session_state"):
        st.session_state.api_keys_data = data
    return key_id


def update_api_key(key_id: str, name: str | None = None, key: str | None = None, provider: str | None = None):
    """Mevcut bir API anahtarının bilgilerini günceller ve kalıcı dosyaya yazar."""
    data = get_api_keys_data()
    k = next((item for item in data.get("keys", []) if item.get("id") == key_id), None)
    if not k:
        return
    if name is not None and name.strip():
        k["name"] = name.strip()
    if key is not None and key.strip():
        k["key"] = key.strip()
    if provider is not None and provider.strip():
        k["provider"] = provider.strip()

    save_keys_to_disk(data)
    if hasattr(st, "session_state"):
        st.session_state.api_keys_data = data


def delete_api_key(key_id: str):
    """API anahtarını siler, varsayılanlardan temizler ve kalıcı dosyaya yazar."""
    data = get_api_keys_data()
    keys = data.get("keys", [])
    data["keys"] = [k for k in keys if k.get("id") != key_id]

    default_keys = data.get("default_keys", {})
    to_remove = [m for m, kid in default_keys.items() if kid == key_id]
    for m in to_remove:
        default_keys.pop(m, None)

    save_keys_to_disk(data)

    if hasattr(st, "session_state"):
        st.session_state.api_keys_data = data

        # Oturumdaki aktif seçimleri temizle
        state = st.session_state
        for _, info in MODELS.items():
            mid = info["id"]
            safe_m = sanitize_model_key(mid)
            widget_key = f"active_key_select_{safe_m}"
            if state.get(widget_key) == key_id:
                new_def = get_default_key_id_for_model(mid, info["provider"])
                if new_def:
                    state[widget_key] = new_def
                else:
                    state.pop(widget_key, None)


def sync_api_keys_storage(storage_key: str = "llm_api_keys") -> dict:
    """Kalıcı dosya ile st.session_state arasındaki API anahtarlarını senkronize eder."""
    state = getattr(st, "session_state", None)
    if state is None:
        return load_keys_from_disk()

    if not state.get("keys_initialized", False):
        state.api_keys_data = load_keys_from_disk()
        state.keys_initialized = True

        for _, info in MODELS.items():
            mid = info["id"]
            safe_m = sanitize_model_key(mid)
            select_key = f"active_key_select_{safe_m}"
            if select_key not in state:
                def_k = get_default_key_id_for_model(mid, info["provider"])
                if def_k:
                    state[select_key] = def_k

    return state.api_keys_data


def get_saved_api_key(model: str, provider: str) -> str:
    """Geriye dönük uyumluluk: Verilen model ya da sağlayıcı için aktif/varsayılan API anahtarını döner."""
    state = getattr(st, "session_state", {})
    safe_m = sanitize_model_key(model)
    selected_id = state.get(f"active_key_select_{safe_m}") if hasattr(st, "session_state") else None
    if selected_id:
        if selected_id == "__env__":
            env_var = PROVIDERS.get(provider, {}).get("env")
            return os.environ.get(env_var, "") if env_var else ""
        k_obj = get_key_by_id(selected_id)
        if k_obj:
            return k_obj.get("key", "")

    def_id = get_default_key_id_for_model(model, provider)
    if def_id:
        k_obj = get_key_by_id(def_id)
        if k_obj:
            return k_obj.get("key", "")

    prov_keys = get_all_keys(provider)
    if prov_keys:
        return prov_keys[0].get("key", "")

    return ""


def save_api_key_for_model(model: str, provider: str, key_val: str | None):
    """Geriye dönük uyumluluk: Tekil anahtar ekler veya siler."""
    if key_val and key_val.strip():
        add_api_key(
            name=f"{model} Anahtarı",
            key=key_val.strip(),
            provider=provider,
            default_for_models=[model],
        )
    else:
        def_id = get_default_key_id_for_model(model, provider)
        if def_id:
            delete_api_key(def_id)
