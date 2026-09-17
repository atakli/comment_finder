"""Web arayüzü: YouTube / Ekşi Sözlük / Instagram / Facebook'tan yorum çek, prompt'a göre LLM ile ayıkla.

Ziyaretçiler için sadeleştirilmiş okuma/arama modu, yönetici için tam ayar ve ayıklama paneli.
"""
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import history
from llm_filter import (MODELS, PROVIDERS, TRANSLATE_SUFFIX, build_requests, count_input_tokens, default_backend,
                        estimate_cost, filter_items, get_model_thinking_config,
                        needs_translation_check, strip_translate_suffix,
                        translate_items)
from local_storage import (add_api_key, delete_api_key, get_all_keys,
                           get_default_key_id_for_model, get_key_by_id,
                           get_saved_api_key, mask_key, sanitize_model_key,
                           save_api_key_for_model, set_default_key_for_model,
                           sync_api_keys_storage)
from scrapers import (fetch_comments, fetch_entries, fetch_facebook_comments,
                      fetch_instagram_comments, get_facebook_cookies,
                      get_instagram_cookies)

st.set_page_config(page_title="Yorum Ayıklayıcı", page_icon="🔎", layout="wide")


def load_env_file():
    """Varsa .env dosyasındaki ve Streamlit Secrets'taki ortam değişkenlerini yükler."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
        except OSError:
            pass

    # Streamlit Cloud (Settings -> Secrets) panelinden girilen değişkenleri ortama aktar
    try:
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if isinstance(v, str) and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


load_env_file()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD") or os.environ.get("ADMIN_KEY") or "admin"


# URL'deki ?admin= parametresi ile hızlı yönetici girişi
if "admin" in st.query_params:
    token = (st.query_params.get("admin") or "").strip()
    if token == ADMIN_PASSWORD.strip():
        st.session_state.is_admin = True
        st.session_state.preview_visitor = False
        st.query_params.pop("admin", None)
        st.toast("🔑 Yönetici girişi yapıldı!", icon="✅")
    else:
        if "#" in ADMIN_PASSWORD and ADMIN_PASSWORD.startswith(token):
            st.toast("⚠️ Şifredeki '#' karakteri URL'de kesiliyor! Girişi pencereden yapın veya '#' yerine '%23' yazın.", icon="⚠️")
        else:
            st.toast("❌ Hatalı yönetici şifresi!", icon="⚠️")
        st.query_params.pop("admin", None)


@st.dialog("Yönetici Girişi")
def admin_login_dialog():
    st.write("Yönetici paneline, ayıklama ayarlarına ve veri çekme aracına erişmek için şifrenizi girin.")
    with st.form("admin_login_form", clear_on_submit=False):
        pwd = st.text_input("Yönetici Şifresi", type="password", key="admin_pwd_dialog_input")
        c1, c2 = st.columns(2)
        submit = c1.form_submit_button("Giriş Yap", type="primary", use_container_width=True)
        cancel = c2.form_submit_button("Vazgeç", use_container_width=True)
        if submit:
            if (pwd or "").strip() == ADMIN_PASSWORD.strip():
                st.session_state.is_admin = True
                st.session_state.preview_visitor = False
                st.toast("🔑 Yönetici girişi başarılı!", icon="✅")
                st.rerun()
            else:
                st.error("Hatalı şifre!")
        if cancel:
            st.rerun()


def admin_logout():
    st.session_state.is_admin = False
    st.session_state.preview_visitor = False
    st.query_params.pop("run", None)
    st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_youtube(url, max_comments, include_replies):
    return fetch_comments(url, max_comments, include_replies)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_eksi(url, max_pages, nice):
    return fetch_entries(url, max_pages, nice)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_instagram(url, max_comments, include_replies=True, sessionid=None, apify_token=None):
    return fetch_instagram_comments(url, max_comments, include_replies, sessionid=sessionid, apify_token=apify_token)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_facebook(url, max_comments, cookies=None, access_token=None, apify_token=None):
    return fetch_facebook_comments(url, max_comments, cookies=cookies, access_token=access_token, apify_token=apify_token)


def to_df(rows, cols):
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# Türkçe büyük/küçük harf ve i/ı farkını yok say (karakter sayısı korunur, vurgulama konumları kaymaz)
_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def fold(text):
    text = str(text or "").translate(_FOLD)
    folded = text.lower()
    if len(folded) == len(text):
        return folded
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def matches(item, query, fields=("text", "group", "author", "title")):
    return fold(query) in fold(" ".join(str(item.get(f) or "") for f in fields))


def md_escape(text):
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text)


def highlight(text, query):
    """Markdown güvenli metin; aranan ifade vurgulu, satır sonları korunur."""
    text, q = str(text or ""), fold(query.strip())
    parts, pos = [], 0
    if q:
        folded = fold(text)
        if len(folded) == len(text):
            for m in re.finditer(re.escape(q), folded):
                parts.append(md_escape(text[pos:m.start()]))
                parts.append(f":orange[**{md_escape(text[m.start():m.end()])}**]")
                pos = m.end()
    parts.append(md_escape(text[pos:]))
    return "".join(parts).replace("\n", "  \n")


SOURCE_ICONS = {
    "youtube": "📺",
    "eksi": "🟢",
    "instagram": "📷",
    "facebook": "📘",
}


def get_source_icon(source: str | None) -> str:
    s = str(source or "").lower().replace(" ", "").replace("sözlük", "")
    for k, v in SOURCE_ICONS.items():
        if k in s:
            return v
    return "🔗"


def clear_search_query(key):
    st.session_state[key] = ""



def inject_keyboard_nav():
    """Masaüstünde Sağ/Sol ok tuşları ve mobilde yatay kaydırma ile gezinme betiği."""
    st.html("""
    <style>
    /* Mobil okuma & kart düzeni iyileştirmeleri */
    .stContainer > div {
        word-break: break-word;
    }
    @media (max-width: 768px) {
        button {
            min-height: 42px !important;
        }
        p, div {
            font-size: 1.01rem !important;
            line-height: 1.55 !important;
        }
    }
    </style>
    <script>
    (function() {
        const doc = (window.parent && window.parent.document) ? window.parent.document : document;
        const win = window.parent || window;

        if (win.__commentNavInitialized) return;
        win.__commentNavInitialized = true;

        let lastTrigger = 0;
        function getButtons() {
            const btns = Array.from(doc.querySelectorAll('button'));
            const prev = btns.find(b => b.offsetParent !== null && b.innerText && (b.innerText.includes('Önceki') || b.innerText.includes('◀')));
            const next = btns.find(b => b.offsetParent !== null && b.innerText && (b.innerText.includes('Sonraki') || b.innerText.includes('▶')));
            return { prev, next };
        }

        doc.addEventListener('keydown', function(e) {
            const active = doc.activeElement;
            if (active && (
                active.tagName === 'INPUT' ||
                active.tagName === 'TEXTAREA' ||
                active.isContentEditable ||
                active.closest('[data-baseweb="input"]') ||
                active.closest('[data-baseweb="textarea"]')
            )) {
                return;
            }

            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                const now = Date.now();
                if (now - lastTrigger < 250) return;
                const { prev, next } = getButtons();
                if (e.key === 'ArrowLeft' && prev && !prev.disabled) {
                    e.preventDefault();
                    lastTrigger = now;
                    prev.click();
                } else if (e.key === 'ArrowRight' && next && !next.disabled) {
                    e.preventDefault();
                    lastTrigger = now;
                    next.click();
                }
            }
        });

        let startX = 0, startY = 0;
        doc.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            }
        }, {passive: true});

        doc.addEventListener('touchend', function(e) {
            if (!startX || !startY) return;
            const diffX = e.changedTouches[0].clientX - startX;
            const diffY = e.changedTouches[0].clientY - startY;
            startX = 0; startY = 0;
            if (Math.abs(diffX) > 75 && Math.abs(diffX) > Math.abs(diffY) * 1.8) {
                const now = Date.now();
                if (now - lastTrigger < 300) return;
                const { prev, next } = getButtons();
                if (diffX > 0 && prev && !prev.disabled) {
                    lastTrigger = now;
                    prev.click();
                } else if (diffX < 0 && next && !next.disabled) {
                    lastTrigger = now;
                    next.click();
                }
            }
        }, {passive: true});
    })();
    </script>
    """, unsafe_allow_javascript=True)


def item_browser(rows, cols, column_config, key, query):
    """Tablo, kart akışı ve tekil odak inceleme modları ile gelişmiş gezinme."""
    if not rows:
        st.info("Eşleşen öğe bulunamadı.")
        return

    inject_keyboard_nav()

    curr_idx_key = f"curr_idx_{key}"
    slider_key = f"{key}_slider"
    df_key = f"df_{key}"

    if curr_idx_key not in st.session_state:
        st.session_state[curr_idx_key] = 0
    st.session_state[curr_idx_key] = max(0, min(st.session_state[curr_idx_key], len(rows) - 1))
    idx = st.session_state[curr_idx_key]

    vm_col, _ = st.columns([2, 3])
    with vm_col:
        view_mode = st.segmented_control(
            "Görünüm Seçeneği",
            options=["🃏 Kartlar", "🔍 Tekil İnceleme", "📊 Tablo"],
            default="🃏 Kartlar",
            key=f"vmode_{key}",
            label_visibility="collapsed"
        ) or "🃏 Kartlar"

    def step_idx(delta):
        new_idx = max(0, min(len(rows) - 1, st.session_state[curr_idx_key] + delta))
        st.session_state[curr_idx_key] = new_idx
        if slider_key in st.session_state:
            st.session_state[slider_key] = new_idx + 1
        if df_key in st.session_state:
            st.session_state[df_key] = {"selection": {"rows": [new_idx], "columns": []}}

    def on_slider_change():
        new_idx = st.session_state[slider_key] - 1
        st.session_state[curr_idx_key] = new_idx
        if df_key in st.session_state:
            st.session_state[df_key] = {"selection": {"rows": [new_idx], "columns": []}}

    if view_mode == "🔍 Tekil İnceleme":
        if slider_key not in st.session_state:
            st.session_state[slider_key] = idx + 1

        # Üst gezinme çubuğu
        col_prev, col_info, col_next = st.columns([1, 2, 1], vertical_alignment="center")
        with col_prev:
            st.button("◀ Önceki", key=f"{key}_top_prev", disabled=(idx <= 0),
                      use_container_width=True, on_click=step_idx, args=(-1,),
                      help="Kısayol: Sol ok tuşu (←) veya parmakla sağa kaydırma")
        with col_info:
            st.markdown(f"<div style='text-align:center; font-weight:bold;'>Öğe {idx + 1} / {len(rows)}</div>",
                        unsafe_allow_html=True)
            if len(rows) > 1:
                st.slider("Öğe Seçici", min_value=1, max_value=len(rows), key=slider_key,
                          on_change=on_slider_change, label_visibility="collapsed")
        with col_next:
            st.button("Sonraki ▶", key=f"{key}_top_next", disabled=(idx >= len(rows) - 1),
                      use_container_width=True, on_click=step_idx, args=(1,),
                      help="Kısayol: Sağ ok tuşu (→) veya parmakla sola kaydırma")

        # Öğe detay kartı
        it = rows[idx]
        with st.container(border=True):
            badges = [f"**👤 {md_escape(str(it.get('author') or '?'))}**"]
            if it.get("likes") is not None:
                badges.append(f"👍 {it.get('likes')}")
            if it.get("date"):
                badges.append(f"📅 {md_escape(str(it.get('date')))}")
            if it.get("score") is not None:
                badges.append(f"⭐ **{it['score']}/10**")
            if it.get("group"):
                badges.append(f"🏷️ **{md_escape(str(it['group']))}**")
            if it.get("is_reply"):
                badges.append("↪️ yanıt")
            if it.get("translated"):
                badges.append("🌐 Türkçeye çevrildi")
            st.markdown(" · ".join(badges))
            st.divider()
            st.markdown(highlight(it.get("text"), query))
            st.divider()
            b1, b2 = st.columns([3, 1], vertical_alignment="center")
            with b1:
                st.caption(f"{get_source_icon(it.get('source'))} {md_escape(str(it.get('title') or ''))}")
            with b2:
                if it.get("link"):
                    st.link_button("Kaynağında aç ↗", it["link"], use_container_width=True)

        # Alt gezinme çubuğu (uzun metinlerde aşağı kaydırdıktan sonra yukarı çıkma zahmetini önler)
        b_prev, b_info, b_next = st.columns([1, 2, 1], vertical_alignment="center")
        with b_prev:
            st.button("◀ Önceki ", key=f"{key}_bot_prev", disabled=(idx <= 0),
                      use_container_width=True, on_click=step_idx, args=(-1,))
        with b_info:
            st.caption(f"<div style='text-align:center;'>{idx + 1} / {len(rows)} (Klavye: ← / →)</div>",
                       unsafe_allow_html=True)
        with b_next:
            st.button("Sonraki ▶ ", key=f"{key}_bot_next", disabled=(idx >= len(rows) - 1),
                      use_container_width=True, on_click=step_idx, args=(1,))

    elif view_mode == "🃏 Kartlar":
        # Mobil ve akıcı okuma için tam metin kartlar listesi
        page_size = 15
        total_pages = max(1, (len(rows) + page_size - 1) // page_size)
        page_key = f"page_{key}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        st.session_state[page_key] = max(1, min(st.session_state[page_key], total_pages))
        page = st.session_state[page_key]

        def step_page(delta):
            st.session_state[page_key] = max(1, min(total_pages, st.session_state[page_key] + delta))

        start_i = (page - 1) * page_size
        end_i = min(len(rows), start_i + page_size)
        page_items = rows[start_i:end_i]

        if total_pages > 1:
            p1, p2, p3 = st.columns([1, 2, 1], vertical_alignment="center")
            with p1:
                st.button("◀ Önceki Sayfa", key=f"{key}_page_prev", disabled=(page <= 1),
                          use_container_width=True, on_click=step_page, args=(-1,))
            with p2:
                st.markdown(f"<div style='text-align:center; font-weight:500;'>Sayfa {page} / {total_pages} ({len(rows)} öğe)</div>",
                            unsafe_allow_html=True)
            with p3:
                st.button("Sonraki Sayfa ▶", key=f"{key}_page_next", disabled=(page >= total_pages),
                          use_container_width=True, on_click=step_page, args=(1,))

        for i, it in enumerate(page_items, start=start_i + 1):
            with st.container(border=True):
                badges = [f"**#{i} · 👤 {md_escape(str(it.get('author') or '?'))}**"]
                if it.get("likes") is not None:
                    badges.append(f"👍 {it.get('likes')}")
                if it.get("date"):
                    badges.append(f"📅 {md_escape(str(it.get('date')))}")
                if it.get("score") is not None:
                    badges.append(f"⭐ **{it['score']}/10**")
                if it.get("group"):
                    badges.append(f"🏷️ **{md_escape(str(it['group']))}**")
                if it.get("is_reply"):
                    badges.append("↪️ yanıt")
                if it.get("translated"):
                    badges.append("🌐 Çevrildi")
                st.markdown(" · ".join(badges))
                st.markdown(highlight(it.get("text"), query))
                st.divider()
                c_t, c_b = st.columns([4, 1], vertical_alignment="center")
                with c_t:
                    st.caption(f"{get_source_icon(it.get('source'))} {md_escape(str(it.get('title') or ''))}")
                with c_b:
                    if it.get("link"):
                        st.link_button("Aç ↗", it["link"], use_container_width=True)

        if total_pages > 1:
            bp1, bp2, bp3 = st.columns([1, 2, 1], vertical_alignment="center")
            with bp1:
                st.button("◀ Önceki Sayfa ", key=f"{key}_bot_page_prev", disabled=(page <= 1),
                          use_container_width=True, on_click=step_page, args=(-1,))
            with bp2:
                st.caption(f"<div style='text-align:center;'>Sayfa {page} / {total_pages}</div>",
                           unsafe_allow_html=True)
            with bp3:
                st.button("Sonraki Sayfa ▶ ", key=f"{key}_bot_page_next", disabled=(page >= total_pages),
                          use_container_width=True, on_click=step_page, args=(1,))

    else:  # 📊 Tablo
        table, detail = st.columns([3, 2])
        if df_key not in st.session_state:
            st.session_state[df_key] = {"selection": {"rows": [idx], "columns": []}}

        with table:
            event = st.dataframe(
                to_df(rows, cols),
                width="stretch",
                hide_index=True,
                column_config=column_config,
                on_select="rerun",
                selection_mode="single-row",
                key=df_key
            )
            picked = event.selection.rows
            if picked and picked[0] < len(rows) and picked[0] != idx:
                st.session_state[curr_idx_key] = picked[0]
                idx = picked[0]
                if slider_key in st.session_state:
                    st.session_state[slider_key] = picked[0] + 1
                st.rerun()

        with detail, st.container(border=True):
            dnav1, dnav2, dnav3 = st.columns([1, 2, 1], vertical_alignment="center")
            with dnav1:
                st.button("◀ Önceki", key=f"{key}_tab_prev", disabled=(idx <= 0),
                          use_container_width=True, on_click=step_idx, args=(-1,))
            with dnav2:
                st.caption(f"<div style='text-align:center;'><b>{idx + 1} / {len(rows)}</b></div>",
                           unsafe_allow_html=True)
            with dnav3:
                st.button("Sonraki ▶", key=f"{key}_tab_next", disabled=(idx >= len(rows) - 1),
                          use_container_width=True, on_click=step_idx, args=(1,))

            it = rows[idx]
            info = [f"**{md_escape(str(it.get('author') or '?'))}**", f"👍 {it.get('likes', 0)}",
                    md_escape(str(it.get("date") or ""))]
            if it.get("is_reply"):
                info.append("↪️ yanıt")
            if it.get("translated"):
                info.append("🌐 Türkçeye çevrildi")
            if it.get("score") is not None:
                info.append(f"⭐ {it['score']}/10")
            if it.get("group"):
                info.append(f"🏷️ {md_escape(str(it['group']))}")
            st.markdown(" · ".join(x for x in info if x))
            st.divider()
            st.markdown(highlight(it.get("text"), query))
            st.divider()
            st.caption(f"{get_source_icon(it.get('source'))} {md_escape(str(it.get('title') or ''))}")
            if it.get("link"):
                st.link_button("Kaynağında aç ↗", it["link"])


def load_run_into_state(run_id):
    """Geçmişten bir çalıştırmayı yükler (on_click: widget'lar çizilmeden önce çalışır)."""
    meta, data = history.load_run(run_id)
    state = st.session_state
    state.fetched, state.results, state.errors = data["fetched"], data["results"], data["errors"]
    state.run_id, state.run_filtered = run_id, data["results"] is not None
    state.run_source = meta["source"]
    state.source, state.links_text = meta["source"], "\n".join(meta["links"])
    state.criteria = strip_translate_suffix(meta["criteria"])  # sabit ek tekrar yazılmasın
    state.active_criteria = meta.get("criteria", "")
    label = next((k for k, v in MODELS.items() if v["id"] == meta["model"]), None)
    if not label and meta.get("model") in ("gemini-3.1-flash-preview", "gemini-3.1-flash-lite-preview", "models/gemini-3.1-flash-lite-preview"):
        label = "Gemini 3.1 Flash-Lite (Preview)"
    if not label and meta.get("model") in ("gemini-3.8-flash", "models/gemini-3.8-flash"):
        label = "Gemini 3.8 Flash"
    if label:
        state.model_label = label
    if meta.get("thinking_level") and meta.get("model"):
        state[f"thinking_level_{sanitize_model_key(meta['model'])}"] = meta["thinking_level"]


@st.dialog("Kaydı sil")
def confirm_delete(meta):
    st.write(f"**{meta['saved_at']} · {meta['source']}** kaydı silinsin mi?")
    st.caption(f"{(meta['titles'] or meta['links'] or ['?'])[0][:80]} — {meta['n_items']} öğe"
               + ("" if meta["n_selected"] is None else f", {meta['n_selected']} seçili"))
    st.caption("Kayıt çöp kutusuna (`data/trash/`) taşınır, kenar çubuğundan geri alınabilir.")
    y, n = st.columns(2)
    if y.button("Evet, sil", type="primary", width="stretch"):
        state = st.session_state
        history.delete_run(meta["id"])
        was_open = state.get("run_id") == meta["id"]
        if was_open:
            state.run_id = None
        state.deleted_runs.append({"id": meta["id"], "label": f"{meta['saved_at']} · {meta['source']}",
                                   "was_open": was_open})
        st.rerun()
    if n.button("Vazgeç", width="stretch"):
        st.rerun()


def undo_delete():
    state = st.session_state
    last = state.deleted_runs.pop()
    if history.restore_run(last["id"]):
        if last["was_open"] and state.get("run_id") is None:
            state.run_id = last["id"]
    else:
        state.undo_error = f"{last['label']} geri alınamadı (çöp kutusunda yok ya da aynı kayıt zaten var)."


def save_current_run(criteria, model, backend, thinking_level=None):
    state = st.session_state
    try:
        history.save_run(state.run_id, source=state.run_source, criteria=criteria, model=model,
                         backend=backend, fetched=state.fetched, results=state.results, errors=state.errors,
                         thinking_level=thinking_level)
    except OSError as e:
        st.warning(f"Geçmişe kaydedilemedi: {e}")


def get_run_display_title(meta):
    """Kayıt için kullanıcı dostu başlık üretir."""
    criteria = (meta.get("criteria") or "").strip()
    if criteria:
        cleaned = strip_translate_suffix(criteria).strip()
        first_line = (cleaned or criteria).splitlines()[0].strip()
        if first_line:
            return first_line[0].upper() + first_line[1:]
    titles = meta.get("titles") or meta.get("links") or []
    if titles and titles[0]:
        return titles[0]
    return "İnceleme / Çalışma"


# ==============================================================================
# ZİYARETÇİ / OKUYUCU GÖRÜNÜMÜ (Ayar yok, sadece seçilmiş sonuçları arama & okuma)
# ==============================================================================
def render_visitor_view(preview_mode=False):
    if preview_mode:
        st.info("👁️ **Ziyaretçi Önizleme Modundasınız.** Dışarıdan bağlanan ziyaretçiler yalnızca bu arayüzü görür; hiçbir ayara veya API anahtarına erişemez. Yönetici moduna dönmek için sol kenar çubuğundaki önizleme anahtarını kapatabilirsiniz.")

    curated_runs = [r for r in history.list_runs() if r.get("n_selected") and r["n_selected"] > 0]
    if not curated_runs:
        st.title("🔎 Yorum Arşivi")
        st.caption("YouTube, Ekşi Sözlük, Instagram ve Facebook'tan yapay zeka ile derlenmiş seçkin yorumlar ve deneyimler.")
        st.info("Henüz seçilmiş yorum içeren yayınlanmış bir kayıt bulunmuyor.")
        st.divider()
        if not st.session_state.get("is_admin"):
            if st.button("🔑 Yönetici Girişi", key="vis_empty_admin_btn"):
                admin_login_dialog()
        return

    run_options = {}
    for r in curated_runs:
        t = get_run_display_title(r)
        short_t = t if len(t) <= 80 else t[:77] + "..."
        run_options[r["id"]] = f"📌 {short_t} ({r['n_selected']} seçilmiş · {r.get('source', '')})"

    curated_ids = list(run_options)
    wanted = st.query_params.get("run")
    if wanted and wanted not in curated_ids:
        st.warning("Belirtilen çalışma bulunamadı veya yayınlanmış yorum içermiyor.")
        st.query_params.pop("run", None)
        wanted = None

    # URL'de doğrudan bir çalışma seçilmemişse Ana Sayfa (Liste) görünümünü göster
    if not wanted:
        st.title("🔎 Yorum Arşivi")
        st.caption("YouTube, Ekşi Sözlük, Instagram ve Facebook'tan yapay zeka ile derlenmiş seçkin yorumlar ve deneyimler.")

        st.markdown("### 📚 İncelenen Konular ve Çalışmalar")
        st.write("Aşağıdaki listeden incelemek istediğiniz çalışmayı seçebilirsiniz:")

        # Hızlı seçim kutusu (Dropdown)
        selected_from_dropdown = st.selectbox(
            "Çalışma Seçin",
            curated_ids,
            format_func=run_options.get,
            index=None,
            placeholder="🔍 Listeden bir çalışma seçin...",
            key="vis_home_select",
            label_visibility="collapsed"
        )
        if selected_from_dropdown:
            st.query_params["run"] = selected_from_dropdown
            st.rerun()

        st.write("")

        # Kartlar halinde çalışma listesi
        for r in curated_runs:
            with st.container(border=True):
                c_info, c_btn = st.columns([4, 1], vertical_alignment="center")
                with c_info:
                    title = get_run_display_title(r)
                    st.markdown(f"#### 📌 {title}")
                    meta_info = [
                        f"🏷️ **Kaynak:** {r.get('source', '')}",
                        f"⭐ **{r['n_selected']} seçilmiş yorum**",
                        f"📊 Toplam {r.get('n_items', r['n_selected']):,} yorum tarandı",
                        f"🗓️ {r.get('saved_at', '')}"
                    ]
                    st.caption(" · ".join(meta_info))

                    titles = r.get("titles") or r.get("links") or []
                    if titles:
                        with st.expander(f"📺 İncelenen Kaynaklar ({len(titles)})", expanded=False):
                            for t in titles:
                                st.write(f"- {t}")
                with c_btn:
                    if st.button("İncele ➔", key=f"btn_run_{r['id']}", type="primary", use_container_width=True):
                        st.query_params["run"] = r["id"]
                        st.rerun()

        # Alt bilgi & Yönetici girişi
        st.divider()
        f1, f2 = st.columns([4, 1], vertical_alignment="center")
        with f1:
            st.caption("Yorum Ayıklayıcı · Ziyaretçi Okuma Modu")
        with f2:
            if not st.session_state.get("is_admin"):
                if st.button("🔑 Yönetici Girişi", key="vis_home_footer_login"):
                    admin_login_dialog()
        return

    # Tekil bir çalışma inceleniyorsa
    selected_run_id = wanted
    st.query_params["run"] = selected_run_id

    try:
        meta, data = history.load_run(selected_run_id)
    except Exception as e:
        st.error(f"Kayıt yüklenemedi: {e}")
        if st.button("⬅️ Listeye Dön", key="vis_err_back"):
            st.query_params.pop("run", None)
            st.rerun()
        return

    results = data.get("results") or []

    # Üst gezinme çubuğu (Listeye dön & Diğer çalışmaya hızlı geçiş)
    nav_c1, nav_c2 = st.columns([1, 2], vertical_alignment="center")
    with nav_c1:
        if st.button("⬅️ Tüm Çalışmalar", key="vis_top_back_btn", use_container_width=True):
            st.query_params.pop("run", None)
            st.rerun()
    if len(curated_runs) > 1:
        with nav_c2:
            switch_to = st.selectbox(
                "Diğer Çalışmaya Geç",
                curated_ids,
                index=curated_ids.index(selected_run_id),
                format_func=run_options.get,
                key="vis_switch_run_select",
                label_visibility="collapsed"
            )
            if switch_to != selected_run_id:
                st.query_params["run"] = switch_to
                st.rerun()

    # Konu başlığı ve kaynaklar
    with st.container(border=True):
        if meta.get("criteria"):
            st.markdown(f"### 🎯 {meta['criteria']}")
        meta_info = [
            f"🏷️ **Kaynak:** {meta.get('source', '')}",
            f"⭐ **{len(results)} seçilmiş yorum** (toplam {meta.get('n_items', len(results))} yorum incelendi)",
            f"🗓️ {meta.get('saved_at', '')}"
        ]
        st.caption(" · ".join(meta_info))

        titles = meta.get("titles") or meta.get("links") or []
        if titles:
            with st.expander(f"{get_source_icon(meta.get('source'))} İncelenen Kaynaklar ({len(titles)})"):
                for t in titles:
                    st.write(f"- {t}")

    # Arama ve grup filtresi
    has_groups = any(r.get("group") for r in results)
    s_col, g_col = st.columns([2, 1] if has_groups else [1, 0.01])
    with s_col:
        s_in, s_btn = st.columns([6, 1] if has_groups else [11, 1], vertical_alignment="bottom", gap="xsmall", wrap=False)
        with s_in:
            query = st.text_input("🔍 Kelime ile ara", placeholder="Metin, grup, yazar veya başlıkta ara...",
                                  key=f"vis_query_{selected_run_id}", type="search").strip()
        with s_btn:
            st.button("", icon=":material/close:", key=f"vis_clear_{selected_run_id}",
                      help="Aramayı sıfırla", disabled=not bool(query),
                      on_click=clear_search_query, args=(f"vis_query_{selected_run_id}",),
                      width="stretch")

    group_filter = "Tümü"
    groups = sorted({r["group"] for r in results if r.get("group")})
    if groups:
        with g_col:
            group_filter = st.selectbox("Gruba göre süz", ["Tümü"] + groups, key=f"vis_group_{selected_run_id}")
        counts = Counter(r["group"] for r in results if r.get("group"))
        st.caption("Gruplar: " + " · ".join(f"{g} ({c})" for g, c in counts.most_common()))

    shown = [r for r in results
             if (not query or matches(r, query)) and (group_filter == "Tümü" or r.get("group") == group_filter)]

    st.subheader(f"✅ {len(shown)} yorum" + (f" (toplam {len(results)} seçilmiş yorum arasından)" if len(shown) != len(results) else ""))

    link_cfg = {"link": st.column_config.LinkColumn("link", display_text="aç"),
                "text": st.column_config.TextColumn("metin", width="large")}
    cols = ["score", "group", "text", "likes", "author", "date", "title", "link"]
    item_browser(shown, cols, {
        **link_cfg,
        "score": st.column_config.ProgressColumn("puan", min_value=0, max_value=10, format="%d"),
        "group": st.column_config.TextColumn("grup", width="medium")
    }, f"vis_{selected_run_id}_{fold(query)}_{group_filter}", query)

    # İndirme butonları
    df = to_df(results, cols)
    d1, d2 = st.columns(2)
    d1.download_button("📥 Tüm Seçilenleri CSV İndir", df.to_csv(index=False).encode("utf-8-sig"),
                       f"{selected_run_id}_secilenler.csv", "text/csv")
    d2.download_button("📥 Tüm Seçilenleri JSON İndir", df.to_json(orient="records", force_ascii=False, indent=2),
                       f"{selected_run_id}_secilenler.json", "application/json")

    # Alt bilgi & Yönetici girişi
    st.divider()
    f1, f2, f3 = st.columns([1.5, 2.5, 1], vertical_alignment="center")
    with f1:
        if st.button("⬅️ Tüm Çalışmalara Dön", key="vis_footer_back", use_container_width=True):
            st.query_params.pop("run", None)
            st.rerun()
    with f2:
        st.caption("Yorum Ayıklayıcı · Ziyaretçi Okuma Modu")
    with f3:
        if not st.session_state.get("is_admin"):
            if st.button("🔑 Yönetici Girişi", key="vis_footer_login", use_container_width=True):
                admin_login_dialog()


# ==============================================================================
# YÖNETİCİ GÖRÜNÜMÜ (Veri çekme, prompt ayıklama, LLM seçimi ve geçmiş yönetimi)
# ==============================================================================
def render_admin_view():
    state = st.session_state
    state.setdefault("fetched", {})
    state.setdefault("results", None)
    state.setdefault("errors", [])
    state.setdefault("run_id", None)
    state.setdefault("run_filtered", False)
    state.setdefault("exact_tokens", {})
    state.setdefault("deleted_runs", [])
    sync_api_keys_storage()

    if "session_loaded" not in state:
        state.session_loaded = True
        run_ids = [r["id"] for r in history.list_runs()]
        wanted = st.query_params.get("run")
        if run_ids:
            load_run_into_state(wanted if wanted in run_ids else run_ids[0])

    # ---------------- Kenar çubuğu ----------------
    with st.sidebar:
        source = st.radio("Kaynak", ["YouTube", "Ekşi Sözlük", "Instagram", "Facebook"], horizontal=True, key="source")
        state.setdefault("run_source", source)

        st.subheader("Çekme ayarları")
        ig_sessionid = None
        ig_apify_token = None
        fb_cookies = None
        fb_access_token = None
        fb_apify_token = None

        if source == "YouTube":
            max_comments = st.number_input("Video başına en fazla yorum", 50, 20000, 1000, step=100)
            include_replies = st.checkbox("Yanıtları da dahil et", value=True)
        elif source == "Ekşi Sözlük":
            max_pages = st.number_input("Başlık başına en fazla sayfa (sayfa = 10 entry)", 1, 1000, 20)
            nice = st.checkbox("Şükela sıralaması (en çok favorilenenler önce)", value=True)
        elif source == "Instagram":
            max_comments = st.number_input("Gönderi başına en fazla yorum", 10, 5000, 200, step=50)
            include_replies = st.checkbox("Yanıtları da dahil et", value=True)

            auto_ig_cookies, auto_ig_browser = get_instagram_cookies()
            if auto_ig_cookies:
                st.success(f"🟢 **{auto_ig_browser.capitalize()}** tarayıcınızdaki Instagram oturumu otomatik kullanılacak (çerez sormaz).")
            else:
                st.info("ℹ️ Tarayıcınızda (Firefox, Chrome vb.) Instagram'a giriş yaptığınızda sistem çerezi otomatik algılar.")

            with st.expander("⚙️ Manuel / Alternatif Ayarlar"):
                ig_method = st.radio(
                    "Yöntem",
                    ["Tarayıcıdan Otomatik", "Manuel sessionid", "Apify API"],
                    horizontal=True,
                    key="ig_method",
                )
                if ig_method == "Manuel sessionid":
                    ig_sessionid = st.text_input(
                        "Instagram sessionid çerezi",
                        value=os.environ.get("INSTAGRAM_SESSIONID", ""),
                        type="password",
                        key="ig_sessionid_input",
                    ).strip() or None
                elif ig_method == "Apify API":
                    ig_apify_token = st.text_input(
                        "Apify API Token",
                        value=os.environ.get("APIFY_API_KEY", ""),
                        type="password",
                        key="ig_apify_input",
                    ).strip() or None

        elif source == "Facebook":
            max_comments = st.number_input("Gönderi başına en fazla yorum", 10, 5000, 200, step=50)

            auto_fb_cookies, auto_fb_browser = get_facebook_cookies()
            if auto_fb_cookies:
                st.success(f"🟢 **{auto_fb_browser.capitalize()}** tarayıcınızdaki Facebook oturumu otomatik kullanılacak (çerez sormaz).")
            else:
                st.info("ℹ️ Tarayıcınızda (Firefox, Chrome vb.) Facebook'a giriş yaptığınızda sistem oturumu otomatik algılar.")

            with st.expander("⚙️ Manuel / Alternatif Ayarlar"):
                fb_method = st.radio(
                    "Yöntem",
                    ["Tarayıcıdan Otomatik", "Apify API (Önerilen)", "Facebook Access Token (Graph API)", "Manuel Çerez"],
                    key="fb_method",
                )
                if fb_method == "Apify API (Önerilen)":
                    fb_apify_token = st.text_input(
                        "Apify API Token",
                        value=os.environ.get("APIFY_API_KEY", ""),
                        type="password",
                        key="fb_apify_input",
                    ).strip() or None
                elif fb_method == "Facebook Access Token (Graph API)":
                    fb_access_token = st.text_input(
                        "Facebook Access Token",
                        value=os.environ.get("FACEBOOK_ACCESS_TOKEN", ""),
                        type="password",
                        key="fb_token_input",
                    ).strip() or None
                elif fb_method == "Manuel Çerez":
                    fb_cookies = st.text_input(
                        "Facebook Çerezleri",
                        value=os.environ.get("FACEBOOK_COOKIES", ""),
                        type="password",
                        key="fb_cookies_input",
                    ).strip() or None

        st.subheader("Ayıklama (LLM)")
        model_label = st.selectbox("Model", list(MODELS), key="model_label")
        model_info = MODELS[model_label]
        model, provider = model_info["id"], model_info["provider"]

        if provider == "anthropic":
            backends = {"cli": "Claude Code CLI (abonelik)", "api": "Anthropic API (ANTHROPIC_API_KEY)"}
            saved_has_anthropic = bool(get_all_keys("anthropic") or os.environ.get("ANTHROPIC_API_KEY"))
            def_backend = "api" if saved_has_anthropic else default_backend()
            backend_idx = list(backends).index(state.get("anthropic_backend_choice", def_backend))
            backend = st.radio("Backend", list(backends), format_func=backends.get,
                               index=backend_idx, key="anthropic_backend_choice")
        else:
            backend = "api"
            st.caption(f"{PROVIDERS[provider]['label']} modeli doğrudan API ile çağrılır (CLI seçeneği yok).")

        api_key = None
        if backend == "api":
            env_var = PROVIDERS[provider]["env"]
            env_val = os.environ.get(env_var)
            safe_m = sanitize_model_key(model)
            select_key = f"active_key_select_{safe_m}"

            provider_keys = get_all_keys(provider)
            default_key_id = get_default_key_id_for_model(model, provider)

            # Seçenek listesi: kayıtlı anahtarlar + varsa ortam değişkeni
            key_options = []
            lbl_to_kid = {}
            kid_to_lbl = {}

            for k in provider_keys:
                kid = k["id"]
                lbl = f"🔑 {k['name']} ({mask_key(k['key'])})"
                if lbl in key_options:
                    lbl = f"🔑 {k['name']} ({mask_key(k['key'])}) #{kid[-4:]}"
                key_options.append(lbl)
                lbl_to_kid[lbl] = kid
                kid_to_lbl[kid] = lbl

            if env_val:
                env_lbl = f"🌐 Ortam Değişkeni (.env: {mask_key(env_val)})"
                key_options.append(env_lbl)
                lbl_to_kid[env_lbl] = "__env__"
                kid_to_lbl["__env__"] = env_lbl

            if key_options:
                curr_sel_id = state.get(select_key)
                if curr_sel_id in kid_to_lbl:
                    curr_sel_lbl = kid_to_lbl[curr_sel_id]
                elif default_key_id in kid_to_lbl:
                    curr_sel_lbl = kid_to_lbl[default_key_id]
                elif provider_keys:
                    curr_sel_lbl = kid_to_lbl[provider_keys[0]["id"]]
                else:
                    curr_sel_lbl = key_options[0]

                widget_k = f"select_widget_{safe_m}"
                sel_idx = key_options.index(curr_sel_lbl) if curr_sel_lbl in key_options else 0

                def on_key_change(sm=safe_m):
                    lbl = state.get(f"select_widget_{sm}")
                    kid = lbl_to_kid.get(lbl)
                    if kid:
                        state[f"active_key_select_{sm}"] = kid

                selected_label = st.selectbox(
                    f"{PROVIDERS[provider]['label']} API Anahtarı",
                    key_options,
                    index=sel_idx,
                    key=widget_k,
                    on_change=on_key_change,
                )
                selected_key_id = lbl_to_kid.get(selected_label) or state.get(select_key) or default_key_id
                if selected_key_id:
                    state[select_key] = selected_key_id

                is_curr_default = (selected_key_id == default_key_id)
                col_info, col_btn = st.columns([3, 2], vertical_alignment="center")
                with col_info:
                    if is_curr_default:
                        st.caption(f"⭐ **{model_label}** için varsayılan.")
                    else:
                        st.caption("⚪ Varsayılan değil.")
                with col_btn:
                    if not is_curr_default:
                        if st.button("⭐ Varsayılan Yap", key=f"btn_set_def_{safe_m}",
                                     help=f"Seçili anahtarı '{model_label}' için varsayılan yap"):
                            set_default_key_for_model(model, selected_key_id)
                            st.toast(f"⭐ '{selected_label}' {model_label} için varsayılan yapıldı!", icon="⭐")

                if selected_key_id == "__env__":
                    api_key = env_val
                else:
                    k_obj = get_key_by_id(selected_key_id)
                    api_key = k_obj["key"] if k_obj else None
            else:
                st.warning(f"⚠️ {PROVIDERS[provider]['label']} için kayıtlı API anahtarı yok.")
                with st.expander(f"➕ {PROVIDERS[provider]['label']} Anahtarı Ekle", expanded=True):
                    q_name = st.text_input("Anahtar Adı / Etiketi",
                                           placeholder=f"Örn: Kişisel {PROVIDERS[provider]['label']}",
                                           key=f"q_name_{safe_m}")
                    q_val = st.text_input("API Anahtarı", type="password", key=f"q_key_{safe_m}")
                    if st.button("💾 Kaydet ve Bu Modelle Kullan", type="primary", key=f"q_save_{safe_m}", width="stretch"):
                        if not q_name.strip():
                            st.error("Lütfen bir anahtar adı girin.")
                        elif not q_val.strip():
                            st.error("Lütfen API anahtarını girin.")
                        else:
                            new_kid = add_api_key(
                                name=q_name.strip(),
                                key=q_val.strip(),
                                provider=provider,
                                default_for_models=[model],
                            )
                            state[select_key] = new_kid
                            st.toast(f"✅ '{q_name.strip()}' kaydedildi ve varsayılan yapıldı!", icon="🔑")
                            st.rerun()

        key_ready = backend != "api" or bool(api_key)

        # Düşünme parametresi (Örn. Gemini 3.8 Flash, Gemini 3.1 Flash-Lite)
        thinking_cfg = get_model_thinking_config(model)
        thinking_level = None
        if thinking_cfg:
            safe_m = sanitize_model_key(model)
            th_levels = thinking_cfg["levels"]
            th_labels = thinking_cfg["labels"]
            th_def = thinking_cfg["default"]
            th_widget_key = f"thinking_level_{safe_m}"
            th_curr = state.get(th_widget_key, th_def)
            if th_curr not in th_levels:
                th_curr = th_def
            th_idx = th_levels.index(th_curr)

            if thinking_cfg.get("mandatory"):
                st.markdown("**🧠 Düşünme Modu (Thinking)**")
                st.caption("ℹ️ Bu modelde düşünme modu mecburi olarak açıktır; akıl yürütme seviyesini seçebilirsiniz:")
            else:
                st.markdown("**🧠 Düşünme Seviyesi (Thinking)**")
                st.caption("Modelin akıl yürütme derinliğini belirleyin:")

            thinking_level = st.radio(
                "Düşünme Seviyesi",
                th_levels,
                format_func=lambda x: th_labels.get(x, x),
                index=th_idx,
                key=th_widget_key,
                label_visibility="collapsed",
            )

        batch_size = st.number_input("İstek başına yorum sayısı", 20, 500, 150, step=10)
        workers = st.number_input("Paralel istek", 1, 8, 4)

        # ---------------- API Anahtarları Yönetimi ----------------
        all_saved_keys = get_all_keys()
        with st.expander(f"🔑 API Anahtarları ({len(all_saved_keys)} kayıtlı)", expanded=False):
            st.caption("🔒 API anahtarları bu bilgisayarda kalıcı bir dosyada saklanır (`data/api_keys.json`), Git'e veya geçmişe eklenmez.")
            tab_list, tab_defaults, tab_add = st.tabs(["📋 Liste", "🎯 Varsayılanlar", "➕ Yeni Ekle"])

            with tab_list:
                if all_saved_keys:
                    for k in all_saved_keys:
                        k_id = k["id"]
                        k_name = k["name"]
                        k_prov = k["provider"]
                        k_mask = mask_key(k["key"])
                        prov_lbl = PROVIDERS.get(k_prov, {}).get("label", k_prov)

                        def_for = [
                            m_lbl for m_lbl, m_info in MODELS.items()
                            if get_default_key_id_for_model(m_info["id"]) == k_id
                        ]

                        c_text, c_del = st.columns([4, 1], vertical_alignment="center")
                        with c_text:
                            st.markdown(f"**{k_name}** `({prov_lbl})`")
                            def_desc = f"⭐ Varsayılan: {', '.join(def_for)}" if def_for else "⚪ Varsayılan değil"
                            st.caption(f"`{k_mask}` · {def_desc}")
                        with c_del:
                            if st.button("🗑️", key=f"btn_del_key_{k_id}", help=f"'{k_name}' anahtarını sil"):
                                delete_api_key(k_id)
                                st.toast(f"🗑️ '{k_name}' silindi.")
                                st.rerun()
                        st.divider()
                else:
                    st.info("Henüz kayıtlı API anahtarı yok. '➕ Yeni Ekle' sekmesinden ekleyebilirsiniz.")

            with tab_defaults:
                st.caption("Her model seçildiğinde otomatik seçilecek varsayılan anahtarı belirleyin:")
                for m_lbl, m_info in MODELS.items():
                    m_id = m_info["id"]
                    m_prov = m_info["provider"]
                    prov_keys = get_all_keys(m_prov)
                    env_k = os.environ.get(PROVIDERS[m_prov]["env"])

                    def_opts = ["— Varsayılan Yok —"]
                    def_lbl_to_id = {"— Varsayılan Yok —": "none"}
                    def_id_to_lbl = {"none": "— Varsayılan Yok —"}

                    for pk in prov_keys:
                        pk_lbl = f"🔑 {pk['name']} ({mask_key(pk['key'])})"
                        if pk_lbl in def_opts:
                            pk_lbl = f"🔑 {pk['name']} ({mask_key(pk['key'])}) #{pk['id'][-4:]}"
                        def_opts.append(pk_lbl)
                        def_lbl_to_id[pk_lbl] = pk["id"]
                        def_id_to_lbl[pk["id"]] = pk_lbl

                    if env_k:
                        env_lbl = f"🌐 .env ({PROVIDERS[m_prov]['env']})"
                        def_opts.append(env_lbl)
                        def_lbl_to_id[env_lbl] = "__env__"
                        def_id_to_lbl["__env__"] = env_lbl

                    curr_def = get_default_key_id_for_model(m_id)
                    curr_def_lbl = def_id_to_lbl.get(curr_def, "— Varsayılan Yok —")
                    curr_idx = def_opts.index(curr_def_lbl) if curr_def_lbl in def_opts else 0

                    def_change_key = f"model_def_setting_{sanitize_model_key(m_id)}"
                    new_def_lbl = st.selectbox(
                        f"🤖 {m_lbl}",
                        def_opts,
                        index=curr_idx,
                        key=def_change_key,
                    )
                    new_def_id = def_lbl_to_id.get(new_def_lbl, "none")
                    if new_def_id != (curr_def or "none"):
                        set_default_key_for_model(m_id, None if new_def_id == "none" else new_def_id)
                        if new_def_id != "none":
                            state[f"active_key_select_{sanitize_model_key(m_id)}"] = new_def_id
                        st.toast(f"⭐ {m_lbl} için varsayılan güncellendi!")
                        st.rerun()

            with tab_add:
                st.markdown("##### ➕ Yeni Anahtar Kaydet")
                add_name = st.text_input("Anahtar Adı / Etiketi",
                                         placeholder="Örn: Kişisel Claude Hesabım, İş Gemini...",
                                         key="tab_add_key_name")
                add_prov = st.selectbox("Sağlayıcı", list(PROVIDERS),
                                        format_func=lambda p: PROVIDERS[p]["label"],
                                        index=0 if provider == "anthropic" else 1,
                                        key="tab_add_key_prov")
                add_val = st.text_input("API Anahtarı", type="password",
                                        placeholder="sk-ant-... veya AIza...",
                                        key="tab_add_key_val")

                matching_models = [lbl for lbl, info in MODELS.items() if info["provider"] == add_prov]
                def_defaults = [model_label] if model_label in matching_models else matching_models[:1]
                add_def_models = st.multiselect("Varsayılan yapılacak modeller:", matching_models,
                                                default=def_defaults, key="tab_add_key_def_models")

                if st.button("💾 Anahtarı Kaydet", type="primary", key="tab_add_key_btn", width="stretch"):
                    if not add_name.strip():
                        st.error("Lütfen bir anahtar adı girin.")
                    elif not add_val.strip():
                        st.error("Lütfen geçerli bir API anahtarı girin.")
                    else:
                        target_model_ids = [MODELS[lbl]["id"] for lbl in add_def_models]
                        new_kid = add_api_key(
                            name=add_name.strip(),
                            key=add_val.strip(),
                            provider=add_prov,
                            default_for_models=target_model_ids,
                        )
                        st.toast(f"✅ '{add_name.strip()}' başarıyla kaydedildi!", icon="🔑")
                        st.rerun()

    # ---------------- Ana alan ----------------
    st.title("🔎 Yorum Ayıklayıcı (Yönetici)")
    if source == "YouTube":
        placeholder = "https://www.youtube.com/watch?v=...\nhttps://youtu.be/..."
    elif source == "Ekşi Sözlük":
        placeholder = "https://eksisozluk.com/baslik-adi--123456\nveya doğrudan başlık adı"
    elif source == "Instagram":
        placeholder = "https://www.instagram.com/p/...\nhttps://www.instagram.com/reel/..."
    elif source == "Facebook":
        placeholder = "https://www.facebook.com/.../posts/...\nhttps://www.facebook.com/watch/?v=..."
    else:
        placeholder = "https://..."
    links_text = st.text_area("Linkler (her satıra bir tane)", placeholder=placeholder, height=110, key="links_text")
    criteria = st.text_area(
        "Ayıklama prompt'u",
        placeholder=TRANSLATE_SUFFIX,
        height=90,
        key="criteria",
    )
    full_criteria = criteria.strip() if criteria else ""

    # ---------------- Önceden Çekilmiş / Süzülmüş Yorum Listeleri ----------------
    all_saved_runs = history.list_runs()
    fetched_runs = [r for r in all_saved_runs if (r.get("n_items") or 0) > 0]
    filtered_runs = [r for r in all_saved_runs if r.get("n_selected") is not None]

    fetched_labels = {}
    for r in fetched_runs:
        t = (r.get("titles") or r.get("links") or ["?"])[0]
        if len(t) > 40:
            t = t[:37] + "..."
        fetched_labels[r["id"]] = f"{r['saved_at']} · {r['source']} · {r['n_items']:,} yorum · {t}"

    filtered_labels = {}
    for r in filtered_runs:
        t = (r.get("titles") or r.get("links") or ["?"])[0]
        if len(t) > 35:
            t = t[:32] + "..."
        crit_snip = f" · \"{r.get('criteria', '')[:25]}...\"" if r.get("criteria") else ""
        filtered_labels[r["id"]] = f"{r['saved_at']} · {r['source']} · {r['n_selected']:,} seçili ({r['n_items']:,} içinden) · {t}{crit_snip}"

    fetched_ids = [None] + [r["id"] for r in fetched_runs]
    filtered_ids = [None] + [r["id"] for r in filtered_runs]

    if state.get("target_fetched_id") not in fetched_ids:
        state["target_fetched_id"] = None
    if state.get("target_filtered_id") not in filtered_ids:
        state["target_filtered_id"] = None

    c_drop1, c_drop2 = st.columns(2)
    with c_drop1:
        target_fetched_id = st.selectbox(
            "📥 Önceden çekilmiş yorumlar (üzerine ekle)",
            options=fetched_ids,
            index=fetched_ids.index(state.get("target_fetched_id")),
            format_func=lambda x: "➕ Yeni liste oluştur (ekleme yapma)" if x is None else fetched_labels.get(x, x),
            key="target_fetched_id",
            help="Bir kayıt seçerseniz, yeni çekilen yorumlar sıfırdan liste açmak yerine bu kaydın çekilmiş yorum listesine eklenir.",
        )
        if target_fetched_id:
            sel_meta = next((r for r in fetched_runs if r["id"] == target_fetched_id), None)
            if sel_meta:
                c_info, c_btn = st.columns([3, 1], vertical_alignment="center")
                with c_info:
                    st.caption(f"📌 **Hedef:** {sel_meta['saved_at']} · {sel_meta['n_items']:,} yorum · {len(sel_meta.get('links', []))} link")
                with c_btn:
                    if state.get("run_id") != target_fetched_id:
                        if st.button("👁️ Yükle", key="btn_load_target_fetched", help="Bu kaydın verilerini ekrana yükler", use_container_width=True):
                            load_run_into_state(target_fetched_id)
                            st.rerun()

    with c_drop2:
        target_filtered_id = st.selectbox(
            "🎯 Önceden süzülmüş yorumlar (üzerine ekle)",
            options=filtered_ids,
            index=filtered_ids.index(state.get("target_filtered_id")),
            format_func=lambda x: "➕ Yeni liste oluştur (ekleme yapma)" if x is None else filtered_labels.get(x, x),
            key="target_filtered_id",
            help="Bir kayıt seçerseniz, yeni ayıklanan sonuçlar bu kaydın seçilenler listesine eklenir.",
        )
        if target_filtered_id:
            sel_f_meta = next((r for r in filtered_runs if r["id"] == target_filtered_id), None)
            if sel_f_meta:
                fc_info, fc_btn = st.columns([3, 1], vertical_alignment="center")
                with fc_info:
                    crit_prev = sel_f_meta.get("criteria", "")[:40]
                    st.caption(f"📌 **Hedef:** {sel_f_meta['saved_at']} · {sel_f_meta['n_selected']:,} seçili" + (f" · *{crit_prev}...*" if crit_prev else ""))
                with fc_btn:
                    if state.get("run_id") != target_filtered_id:
                        if st.button("👁️ Yükle", key="btn_load_target_filtered", help="Bu kaydın verilerini ekrana yükler", use_container_width=True):
                            load_run_into_state(target_filtered_id)
                            st.rerun()

    only_new_filter = True
    if target_filtered_id and state.get("last_fetched_new"):
        only_new_filter = st.checkbox(
            f"⚡ Yalnızca az önce yeni çekilen yorumları ({len(state.last_fetched_new)} yorum) ayıkla ve listeye ekle",
            value=True,
            help="İşaretliyse yalnızca az önce çekilen yeni yorumlar LLM'e gönderilir ve hedef süzülmüş listeye eklenir; eski yorumlar için tekrar token harcanmaz.",
            key="only_new_filter_chk",
        )

    c1, c2, c3 = st.columns(3)
    fetch_clicked = c1.button("1️⃣ Yorumları çek", width="stretch")
    filter_clicked = c2.button("2️⃣ Prompt'a göre ayıkla", type="primary", width="stretch")
    both_clicked = c3.button("⚡ Çek + ayıkla", width="stretch")

    links = [l.strip() for l in links_text.splitlines() if l.strip()]

    if fetch_clicked or both_clicked:
        if not links:
            st.warning("En az bir link girin.")
        else:
            newly_fetched = {}
            fetch_errors = []
            for link in links:
                with st.spinner(f"Çekiliyor: {link}"):
                    try:
                        if source == "YouTube":
                            newly_fetched[link] = cached_youtube(link, int(max_comments), include_replies)
                        elif source == "Ekşi Sözlük":
                            newly_fetched[link] = cached_eksi(link, int(max_pages), nice)
                        elif source == "Instagram":
                            newly_fetched[link] = cached_instagram(
                                link,
                                int(max_comments),
                                include_replies=include_replies,
                                sessionid=ig_sessionid,
                                apify_token=ig_apify_token,
                            )
                        elif source == "Facebook":
                            newly_fetched[link] = cached_facebook(
                                link,
                                int(max_comments),
                                cookies=fb_cookies,
                                access_token=fb_access_token,
                                apify_token=fb_apify_token,
                            )
                    except Exception as e:
                        fetch_errors.append(f"{link} çekilemedi: {e}")
                        st.error(f"{link} çekilemedi: {e}")

            if newly_fetched:
                new_items_count = sum(len(v) for v in newly_fetched.values())
                all_new_items = [it for items in newly_fetched.values() for it in items]
                state.last_fetched_new = all_new_items

                if target_fetched_id:
                    try:
                        base_meta, base_data = history.load_run(target_fetched_id)
                        base_fetched = base_data.get("fetched", {})
                    except Exception as e:
                        st.error(f"Hedef kayıt ({target_fetched_id}) yüklenemedi: {e}")
                        base_meta, base_fetched = {}, {}
                        base_data = {"results": None, "errors": []}

                    merged_fetched, added_count = history.merge_fetched(base_fetched, newly_fetched)
                    state.fetched = merged_fetched
                    state.run_id = target_fetched_id
                    state.run_filtered = (base_data.get("results") is not None)
                    state.run_source = base_meta.get("source", source)
                    state.results = base_data.get("results")
                    state.errors = base_data.get("errors", []) + fetch_errors
                    save_current_run(base_meta.get("criteria", full_criteria), model, backend)
                    st.toast(
                        f"✅ {added_count} yeni yorum '{target_fetched_id}' kaydına eklendi (toplam {sum(len(v) for v in merged_fetched.values())} yorum)!",
                        icon="📥",
                    )
                else:
                    state.fetched = newly_fetched
                    state.results = None
                    state.errors = fetch_errors
                    state.run_id = history.new_run_id()
                    state.run_filtered = False
                    state.run_source = source
                    save_current_run(full_criteria, model, backend)
                    st.toast(f"✅ {new_items_count} yorum çekildi!", icon="📥")

    # ---------------- Maliyet tahmini ----------------
    fetched_items = [(link, items) for link, items in state.fetched.items() if items]
    if not fetched_items and (target_fetched_id or target_filtered_id):
        t_id = target_fetched_id or target_filtered_id
        try:
            _, t_d = history.load_run(t_id)
            fetched_items = [(link, items) for link, items in t_d.get("fetched", {}).items() if items]
        except Exception:
            pass

    if target_filtered_id and state.get("last_fetched_new") and only_new_filter:
        est_items = [("Yeni çekilenler", state["last_fetched_new"])]
    else:
        est_items = fetched_items

    if est_items and criteria.strip():
        requests = [msg for _, items in est_items for _, msg in build_requests(items, full_criteria, int(batch_size))]
        n_total = sum(len(items) for _, items in est_items)
        est_key = (model, criteria, int(batch_size), state.run_id, n_total, thinking_level)
        with st.expander("💰 Ayıklama öncesi maliyet tahmini", expanded=state.results is None):
            if backend == "api" and provider == "anthropic" and st.button("Token'ları API ile tam say (ücretsiz)"):
                try:
                    with st.spinner("Token sayılıyor..."):
                        state.exact_tokens = {est_key: count_input_tokens(model, requests, int(workers), api_key)}
                except Exception as e:
                    st.error(f"Token sayılamadı: {e}")
            est = estimate_cost(model, requests, n_total, state.exact_tokens.get(est_key), thinking_level=thinking_level)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("İstek (parti)", est["batches"], help=f"{n_total} öğe, {est['chars']:,} karakter")
            m2.metric("Giriş token", f"{'' if est['exact'] else '~'}{est['input_tokens']:,}")
            m3.metric("Çıkış token", "{:,}–{:,}".format(*est["output_tokens"]))
            m4.metric(f"Tahmini maliyet ({model_label})", "${:.2f}–${:.2f}".format(*est["cost"]))
            st.caption("🏷️ Bunun içinde gruplama özelliğinin payı: ${:.3f}–${:.3f}".format(*est["group_cost"]))
            notes = ["Giriş tokenı " + ("token sayma API'siyle ölçüldü." if est["exact"]
                                        else "karakter sayısından yaklaşık hesaplandı (~3 karakter/token).")]
            if thinking_cfg and thinking_level:
                notes.append(f"Çıkış aralığı öğelerin %10–50'sinin seçileceği ve '{thinking_cfg['labels'].get(thinking_level, thinking_level)}' düşünme seviyesi varsayar.")
            else:
                notes.append("Çıkış aralığı öğelerin %10–50'sinin seçileceği, seçilen her öğe için indeks/puan/grup "
                             "adı üretileceği ve (Haiku dışında) parti başına düşünme tokenı varsayar.")
            if backend == "cli":
                notes.append("CLI backend'inde doğrudan ücret yok; bu tutar Claude aboneliğinin kullanım "
                             "kotasından düşer (CLI kendi sistem prompt'unu da ekler).")
            if provider != "anthropic":
                notes.append(f"{PROVIDERS[provider]['label']} fiyatlandırması tahminidir, güncel fiyatı sağlayıcıdan "
                             "teyit edin.")
            st.caption(" ".join(notes))

    if (filter_clicked or both_clicked):
        if not state.fetched:
            target_load = target_fetched_id or target_filtered_id
            if target_load:
                try:
                    t_meta, t_data = history.load_run(target_load)
                    state.fetched = t_data.get("fetched", {})
                    state.run_id = target_load
                    state.run_source = t_meta.get("source", source)
                except Exception as e:
                    st.error(f"Seçili kayıt ({target_load}) yüklenemedi: {e}")

        if not state.fetched:
            st.warning("Önce yorumları çekin veya listeden bir kayıt seçin.")
        elif not criteria.strip():
            st.warning("Ayıklama prompt'u girin.")
        elif not key_ready:
            st.warning(f"{PROVIDERS[provider]['label']} için API anahtarı seçilmedi veya girilmedi.")
        else:
            if target_filtered_id and state.get("last_fetched_new") and only_new_filter:
                items_to_filter = state["last_fetched_new"]
                sources = [("Yeni çekilenler", items_to_filter)]
            else:
                sources = [(link, items) for link, items in state.fetched.items() if items]

            results, errors = [], []
            bar = st.progress(0.0, text="Ayıklanıyor...")
            for n, (link, items) in enumerate(sources):
                def progress(done, total, n=n):
                    bar.progress((n + done / total) / len(sources),
                                 text=f"Kaynak {n + 1}/{len(sources)} · parti {done}/{total}")
                picked, errs = filter_items(items, full_criteria, model, backend,
                                            int(batch_size), int(workers), progress, api_key,
                                            thinking_level=thinking_level)
                results += picked
                errors += [f"{link} → {e}" for e in errs]
            bar.empty()

            if target_filtered_id:
                try:
                    filt_meta, filt_data = history.load_run(target_filtered_id)
                    base_results = filt_data.get("results") or []
                    base_fetched = filt_data.get("fetched") or {}
                except Exception as e:
                    st.error(f"Hedef süzülmüş kayıt ({target_filtered_id}) yüklenemedi: {e}")
                    filt_meta, base_results, base_fetched = {}, [], {}
                    filt_data = {"errors": []}

                combined_results, added_res_count = history.merge_results(base_results, results)
                merged_fetched, _ = history.merge_fetched(base_fetched, state.fetched)

                state.fetched = merged_fetched
                state.results = combined_results
                state.errors = filt_data.get("errors", []) + errors
                state.run_id = target_filtered_id
                state.run_filtered = True
                save_criteria = filt_meta.get("criteria") or full_criteria
                state.active_criteria = save_criteria
                save_current_run(save_criteria, model, backend, thinking_level=thinking_level)
                state.last_fetched_new = None
                st.toast(
                    f"✅ {added_res_count} yeni süzülen yorum '{target_filtered_id}' listesine eklendi (toplam {len(combined_results)} seçili)!",
                    icon="🎯",
                )
            else:
                state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
                state.errors = errors
                if state.run_id is None or state.run_filtered:
                    state.run_id = history.new_run_id()
                state.run_filtered = True
                state.active_criteria = full_criteria
                save_current_run(full_criteria, model, backend, thinking_level=thinking_level)
                state.last_fetched_new = None
                st.toast(f"✅ {len(results)} yorum ayıklandı!", icon="🎯")

    # ---------------- Sonuçlar ----------------
    all_items = [it for items in state.fetched.values() for it in items]
    if state.fetched:
        st.caption(" · ".join(f"**{(items[0]['title'] if items else link)[:60]}**: {len(items)}"
                              for link, items in state.fetched.items()) + f" — toplam {len(all_items)} öğe")

    for e in state.errors:
        st.error(e)

    # ---------------- Tekrar Ayıklama (Admine Özel) ----------------
    refilter_clicked = False
    with st.expander("🔄 Farklı Prompt ile Tekrar Ayıkla (Admine Özel)", expanded=bool(state.fetched and state.results)):
        st.markdown("##### 🎯 Yeni / İkincil Prompt ile Yeniden Ayıkla")
        st.caption("Asıl ayıklama prompt'unuzu değiştirmeden, farklı bir kriter girerek mevcut verileri yeniden süzün.")

        refilter_criteria = st.text_area(
            "Tekrar ayıklama prompt'u (asıl prompttan ayrı)",
            placeholder="Örn: Yalnızca garanti, servis veya kronik arıza deneyimi içeren yorumları filtrele...",
            height=85,
            key="refilter_criteria",
        )

        scope_choice = "all"
        if state.results:
            scope_choice = st.radio(
                "Ayıklanacak veri:",
                ["all", "selected"],
                format_func={
                    "all": f"Çekilen tüm yorumlar ({len(all_items)} yorum)",
                    "selected": f"Mevcut seçilenler ({len(state.results)} yorum - sonuçları daralt)",
                }.get,
                horizontal=True,
                key="refilter_scope",
            )

        replace_texts = st.checkbox(
            "Üretilen metinleri öncekilerle değiştir (yeni çalıştırma oluşturma)",
            value=False,
            key="refilter_replace_texts",
            help="Aktif edildiğinde yeni bir çalıştırma oluşturulmaz; LLM'in ürettiği metinler mevcut seçilenlerin üzerine yazılır.",
        )

        if refilter_criteria.strip() and all_items:
            full_ref_crit = refilter_criteria.strip()
            target_list = state.results if (scope_choice == "selected" and state.results) else all_items
            ref_requests = [msg for _, msg in build_requests(target_list, full_ref_crit, int(batch_size))]
            ref_est = estimate_cost(model, ref_requests, len(target_list), thinking_level=thinking_level)
            st.caption(f"💰 Tahmini maliyet ({model_label}): **${ref_est['cost'][0]:.2f}–${ref_est['cost'][1]:.2f}** ({ref_est['batches']} istek, ~{ref_est['input_tokens']:,} token)")

        ref_c1, ref_c2 = st.columns([3, 1], vertical_alignment="center")
        with ref_c1:
            th_note = f" · Düşünme: {thinking_cfg['labels'].get(thinking_level, thinking_level)}" if (thinking_cfg and thinking_level) else ""
            st.caption(f"🤖 Model: **{model_label}**{th_note} · İstek başına {batch_size} yorum · {workers} paralel çalışan")
        with ref_c2:
            btn_label = "🔄 Metinleri Değiştir" if replace_texts else "🔄 Tekrar Ayıkla"
            refilter_clicked = st.button(btn_label, type="primary", key="btn_refilter", width="stretch")

    if refilter_clicked:
        if not refilter_criteria.strip():
            st.warning("Lütfen tekrar ayıklama için bir prompt girin.")
        elif not state.fetched:
            st.warning("Önce yorumları çekin veya geçmişten bir kayıt yükleyin.")
        elif not key_ready:
            st.warning(f"{PROVIDERS[provider]['label']} için API anahtarı seçilmedi veya girilmedi.")
        else:
            full_refilter_criteria = refilter_criteria.strip()
            results, errors = [], []
            bar = st.progress(0.0, text="Yeni prompt'a göre ayıklanıyor...")

            if scope_choice == "selected" and state.results:
                link_groups = {}
                for r in state.results:
                    link_groups.setdefault(r.get("link") or "seçilenler", []).append(r)
                sources = list(link_groups.items())
            else:
                sources = [(link, items) for link, items in state.fetched.items() if items]

            for n, (link, items) in enumerate(sources):
                def progress(done, total, n=n):
                    bar.progress((n + done / total) / len(sources),
                                 text=f"Kaynak {n + 1}/{len(sources)} · parti {done}/{total}")
                picked, errs = filter_items(items, full_refilter_criteria, model, backend,
                                            int(batch_size), int(workers), progress, api_key,
                                            thinking_level=thinking_level)
                results += picked
                errors += [f"{link} → {e}" for e in errs]
            bar.empty()

            if replace_texts:
                new_by_key = {(r.get("source"), str(r.get("id"))): r for r in results}
                updated_count = 0
                if state.results is not None:
                    for it in state.results:
                        k = (it.get("source"), str(it.get("id")))
                        if k in new_by_key:
                            new_it = new_by_key[k]
                            if new_it.get("text") and new_it["text"] != it.get("text"):
                                it["text"] = new_it["text"]
                                updated_count += 1
                            if new_it.get("translated"):
                                it["translated"] = True
                else:
                    state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
                    updated_count = len(state.results)

                if state.fetched:
                    for link, items_list in state.fetched.items():
                        for it in items_list:
                            k = (it.get("source"), str(it.get("id")))
                            if k in new_by_key:
                                new_it = new_by_key[k]
                                if new_it.get("text") and new_it["text"] != it.get("text"):
                                    it["text"] = new_it["text"]
                                if new_it.get("translated"):
                                    it["translated"] = True

                state.errors = errors
                if state.run_id is None:
                    state.run_id = history.new_run_id()
                state.run_filtered = True
                try:
                    old_meta = history.load_run(state.run_id)[0]
                    cur_criteria = old_meta.get("criteria")
                except Exception:
                    cur_criteria = None
                save_criteria = cur_criteria or state.get("active_criteria") or full_refilter_criteria
                save_current_run(save_criteria, model, backend, thinking_level=thinking_level)
                st.toast(f"✅ {updated_count} yorumun metni değiştirildi (mevcut çalıştırma güncellendi)!", icon="🎯")
                st.rerun()
            else:
                state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
                state.errors = errors
                state.run_id = history.new_run_id()
                state.run_filtered = True
                state.active_criteria = full_refilter_criteria
                save_current_run(full_refilter_criteria, model, backend, thinking_level=thinking_level)
                st.toast("✅ Yeni prompt'a göre tekrar ayıklama tamamlandı!", icon="🎯")
                st.rerun()

    link_cfg = {"link": st.column_config.LinkColumn("link", display_text="aç"),
                "text": st.column_config.TextColumn("metin", width="large")}

    query = ""
    if all_items:
        s_in, s_btn = st.columns([11, 1], vertical_alignment="bottom", gap="xsmall", wrap=False)
        with s_in:
            query = st.text_input("🔍 Ara", placeholder="Metin, grup, yazar veya başlıkta ara",
                                  key="search_query", type="search").strip()
        with s_btn:
            st.button("", icon=":material/close:", key="search_query_clear",
                      help="Aramayı sıfırla", disabled=not bool(query),
                      on_click=clear_search_query, args=("search_query",),
                      width="stretch")
    table_key = f"{state.run_id}_{fold(query)}"

    if state.results is not None:
        groups = sorted({r["group"] for r in state.results if r.get("group")})
        group_filter = "Tümü"
        if groups:
            g1, g2 = st.columns([2, 3], vertical_alignment="center")
            with g1:
                group_filter = st.selectbox("Gruba göre süz", ["Tümü"] + groups, key=f"group_filter_{state.run_id}")
            with g2:
                counts = Counter(r["group"] for r in state.results if r.get("group"))
                st.caption(" · ".join(f"{g} ({c})" for g, c in counts.most_common()))
        shown = [r for r in state.results
                 if (not query or matches(r, query)) and (group_filter == "Tümü" or r.get("group") == group_filter)]
        res_table_key = f"{table_key}_{group_filter}"
        if needs_translation_check(state.results):
            t1, t2 = st.columns([3, 1], vertical_alignment="center")
            t1.info("Bu kayıt çeviri özelliğinden önce ayıklanmış; seçilenlerde Türkçe olmayan yorumlar olabilir.", icon="🌐")
            if t2.button("🌐 Türkçe olmayanları çevir", width="stretch"):
                bar = st.progress(0.0, text="Çevriliyor...")
                state.results, errs = translate_items(
                    state.results, model, backend, workers=int(workers), api_key=api_key,
                    thinking_level=thinking_level,
                    on_progress=lambda d, t: bar.progress(d / t, text=f"Çeviri partisi {d}/{t}"))
                bar.empty()
                state.errors = [e for e in state.errors if not e.startswith("Çeviri partisi")] + errs
                try:
                    old = history.load_run(state.run_id)[0]
                    save_current_run(old["criteria"], old["model"], old["backend"], thinking_level=old.get("thinking_level"))
                except (OSError, TypeError):
                    st.warning("Kayıt bulunamadı; çeviri yalnızca bu oturumda geçerli.")
                st.rerun()
        active_c = state.get("active_criteria") or (criteria.strip() if criteria else "")
        if active_c:
            st.caption(f"🎯 **Aktif ayıklama kriteri:** {strip_translate_suffix(active_c)}")
        st.subheader(f"✅ Seçilenler: {len(state.results)} / {len(all_items)}"
                     + (f" · süzülen {len(shown)}" if len(shown) != len(state.results) else ""))
        cols = ["score", "group", "text", "likes", "author", "date", "title", "link"]
        item_browser(shown, cols, {
            **link_cfg, "score": st.column_config.ProgressColumn("puan", min_value=0, max_value=10, format="%d"),
            "group": st.column_config.TextColumn("grup", width="medium")}, f"res_{res_table_key}", query)
        df = to_df(state.results, cols)
        d1, d2 = st.columns(2)
        d1.download_button("CSV indir", df.to_csv(index=False).encode("utf-8-sig"), "secilenler.csv", "text/csv")
        d2.download_button("JSON indir", df.to_json(orient="records", force_ascii=False, indent=2),
                           "secilenler.json", "application/json")

    if all_items:
        shown = [it for it in all_items if matches(it, query)] if query else all_items
        label = f"Tüm çekilen öğeler ({len(all_items)}" + (f", aramada {len(shown)})" if query else ")")
        with st.expander(label, expanded=state.results is None or bool(query)):
            cols = ["text", "likes", "author", "date", "is_reply", "title", "link"]
            item_browser(shown, cols, link_cfg, f"raw_{table_key}", query)
            st.download_button("Ham veriyi CSV indir", to_df(all_items, cols).to_csv(index=False).encode("utf-8-sig"),
                               "tum_yorumlar.csv", "text/csv")

    # ---------------- Geçmiş (kenar çubuğunun sonuna) ----------------
    with st.sidebar:
        st.subheader("🕘 Geçmiş")
        runs = history.list_runs()
        if state.get("undo_error"):
            st.error(state.pop("undo_error"))
        if state.deleted_runs:
            u1, u2 = st.columns([3, 2], vertical_alignment="center")
            u1.caption(f"Silindi: {state.deleted_runs[-1]['label']}")
            u2.button("↩️ Geri al", on_click=undo_delete, width="stretch")
        if not runs:
            st.caption("Henüz kayıt yok. Çekilen veri ve sonuçlar `data/runs/` altına otomatik kaydedilir.")
        else:
            labels = {
                r["id"]: f"{r['saved_at']} · {r['source']} · "
                         f"{'ayıklanmadı' if r['n_selected'] is None else str(r['n_selected']) + ' seçili'}"
                         f"/{r['n_items']} · {(r['titles'] or r['links'] or ['?'])[0][:30]}"
                for r in runs
            }
            ids = list(labels)
            selected_run = st.selectbox("Kayıtlı çalıştırmalar", ids, format_func=labels.get,
                                        index=ids.index(state.run_id) if state.run_id in ids else 0)
            meta = next(r for r in runs if r["id"] == selected_run)
            if meta["criteria"]:
                st.caption(f"Prompt: {meta['criteria'][:200]}")
            h1, h2 = st.columns(2)
            h1.button("Yükle", on_click=load_run_into_state, args=(selected_run,), width="stretch")
            if h2.button("Sil", width="stretch"):
                confirm_delete(meta)

    if state.run_id:
        st.query_params["run"] = state.run_id
    else:
        st.query_params.pop("run", None)


# ==============================================================================
# ANA AKIŞ: Yetki ve Görünüm Belirleme
# ==============================================================================
is_admin = st.session_state.get("is_admin", False)
preview_mode = False

if is_admin:
    with st.sidebar:
        st.markdown("### 🛡️ Yönetici Paneli")
        preview_mode = st.toggle("👁️ Ziyaretçi gözüyle önizle", value=st.session_state.get("preview_visitor", False),
                                 key="toggle_preview_visitor")
        st.session_state.preview_visitor = preview_mode
        if ADMIN_PASSWORD == "admin":
            st.caption("🔒 *Güvenlik İpucu: .env dosyasına `ADMIN_PASSWORD=guclu_sifre` ekleyerek şifrenizi değiştirebilirsiniz.*")
        if st.button("🚪 Çıkış Yap", width="stretch", key="btn_admin_logout"):
            admin_logout()
        st.divider()

if not is_admin or preview_mode:
    render_visitor_view(preview_mode=preview_mode)
else:
    render_admin_view()
