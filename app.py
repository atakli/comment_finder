"""Web arayüzü: YouTube / Ekşi Sözlük'ten yorum çek, prompt'a göre LLM ile ayıkla.

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
                        estimate_cost, filter_items, needs_translation_check, strip_translate_suffix,
                        translate_items, with_translate_suffix)
from scrapers import fetch_comments, fetch_entries

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
    st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_youtube(url, max_comments, include_replies):
    return fetch_comments(url, max_comments, include_replies)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_eksi(url, max_pages, nice):
    return fetch_entries(url, max_pages, nice)


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
                st.caption(f"📺 {md_escape(str(it.get('title') or ''))}")
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
                    st.caption(f"📺 {md_escape(str(it.get('title') or ''))}")
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
            st.caption(md_escape(str(it.get("title") or "")))
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
    label = next((k for k, v in MODELS.items() if v["id"] == meta["model"]), None)
    if not label and meta.get("model") in ("gemini-3.1-flash-preview", "gemini-3.1-flash-lite-preview", "models/gemini-3.1-flash-lite-preview"):
        label = "Gemini 3.1 Flash-Lite (Preview)"
    if label:
        state.model_label = label


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


def save_current_run(criteria, model, backend):
    state = st.session_state
    try:
        history.save_run(state.run_id, source=state.run_source, criteria=criteria, model=model,
                         backend=backend, fetched=state.fetched, results=state.results, errors=state.errors)
    except OSError as e:
        st.warning(f"Geçmişe kaydedilemedi: {e}")


# ==============================================================================
# ZİYARETÇİ / OKUYUCU GÖRÜNÜMÜ (Ayar yok, sadece seçilmiş sonuçları arama & okuma)
# ==============================================================================
def render_visitor_view(preview_mode=False):
    if preview_mode:
        st.info("👁️ **Ziyaretçi Önizleme Modundasınız.** Dışarıdan bağlanan ziyaretçiler yalnızca bu arayüzü görür; hiçbir ayara veya API anahtarına erişemez. Yönetici moduna dönmek için sol kenar çubuğundaki önizleme anahtarını kapatabilirsiniz.")

    st.title("🔎 Yorum Arşivi")
    st.caption("YouTube ve Ekşi Sözlük'ten yapay zeka ile derlenmiş seçkin yorumlar ve deneyimler.")

    curated_runs = [r for r in history.list_runs() if r.get("n_selected") and r["n_selected"] > 0]
    if not curated_runs:
        st.info("Henüz seçilmiş yorum içeren yayınlanmış bir kayıt bulunmuyor.")
        st.divider()
        if not st.session_state.get("is_admin"):
            if st.button("🔑 Yönetici Girişi", key="vis_empty_admin_btn"):
                admin_login_dialog()
        return

    # Çoklu konu varsa konu seçici
    run_options = {
        r["id"]: f"📌 {(r.get('criteria') or (r.get('titles') or ['İnceleme'])[0])[:75]} "
                 f"({r['n_selected']} seçilmiş · {r['source']})"
        for r in curated_runs
    }
    curated_ids = list(run_options)
    wanted = st.query_params.get("run")
    default_idx = curated_ids.index(wanted) if wanted in curated_ids else 0

    if len(curated_runs) > 1:
        selected_run_id = st.selectbox("📖 İncelenen Konuyu Seçin", curated_ids, format_func=run_options.get,
                                       index=default_idx, key="vis_topic_select")
    else:
        selected_run_id = curated_ids[0]

    st.query_params["run"] = selected_run_id

    try:
        meta, data = history.load_run(selected_run_id)
    except Exception as e:
        st.error(f"Kayıt yüklenemedi: {e}")
        return

    results = data.get("results") or []

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
            with st.expander(f"📺 İncelenen Kaynaklar ({len(titles)})"):
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
    f1, f2 = st.columns([4, 1])
    with f1:
        st.caption("Yorum Ayıklayıcı · Ziyaretçi Okuma Modu")
    with f2:
        if not st.session_state.get("is_admin"):
            if st.button("🔑 Yönetici Girişi", key="vis_footer_login"):
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

    if "session_loaded" not in state:
        state.session_loaded = True
        run_ids = [r["id"] for r in history.list_runs()]
        wanted = st.query_params.get("run")
        if run_ids:
            load_run_into_state(wanted if wanted in run_ids else run_ids[0])

    # ---------------- Kenar çubuğu ----------------
    with st.sidebar:
        source = st.radio("Kaynak", ["YouTube", "Ekşi Sözlük"], horizontal=True, key="source")
        state.setdefault("run_source", source)

        st.subheader("Çekme ayarları")
        if source == "YouTube":
            max_comments = st.number_input("Video başına en fazla yorum", 50, 20000, 1000, step=100)
            include_replies = st.checkbox("Yanıtları da dahil et", value=True)
        else:
            max_pages = st.number_input("Başlık başına en fazla sayfa (sayfa = 10 entry)", 1, 1000, 20)
            nice = st.checkbox("Şükela sıralaması (en çok favorilenenler önce)", value=True)

        st.subheader("Ayıklama (LLM)")
        model_label = st.selectbox("Model", list(MODELS), key="model_label")
        model_info = MODELS[model_label]
        model, provider = model_info["id"], model_info["provider"]

        if provider == "anthropic":
            backends = {"cli": "Claude Code CLI (abonelik)", "api": "Anthropic API (ANTHROPIC_API_KEY)"}
            backend = st.radio("Backend", list(backends), format_func=backends.get,
                               index=list(backends).index(default_backend()))
        else:
            backend = "api"
            st.caption(f"{PROVIDERS[provider]['label']} modeli doğrudan API ile çağrılır (CLI seçeneği yok).")

        api_key = None
        if backend == "api":
            env_var = PROVIDERS[provider]["env"]
            if not os.environ.get(env_var):
                api_key = st.text_input(f"{PROVIDERS[provider]['label']} API anahtarı", type="password",
                                        key=f"apikey_{provider}").strip() or None
                if not api_key:
                    st.caption(f"ℹ️ {env_var} tanımlı değil; ayıklamak için API anahtarını girip Enter'a basın.")
        key_ready = backend != "api" or bool(api_key or os.environ.get(PROVIDERS[provider]["env"]))

        batch_size = st.number_input("İstek başına yorum sayısı", 20, 500, 150, step=10)
        workers = st.number_input("Paralel istek", 1, 8, 4)

    # ---------------- Ana alan ----------------
    st.title("🔎 Yorum Ayıklayıcı (Yönetici)")
    placeholder = ("https://www.youtube.com/watch?v=...\nhttps://youtu.be/..." if source == "YouTube"
                   else "https://eksisozluk.com/baslik-adi--123456\nveya doğrudan başlık adı")
    links_text = st.text_area("Linkler (her satıra bir tane)", placeholder=placeholder, height=110, key="links_text")
    criteria = st.text_area("Ayıklama prompt'u",
                            placeholder="Örn: Ürünü uzun süre kullanmış kişilerin somut deneyimleri ve "
                                        "yaşadıkları kronik sorunlar", height=90, key="criteria")
    st.caption(f"➕ Prompt'un sonuna sabit olarak eklenir (tekrar yazmanıza gerek yok): **{TRANSLATE_SUFFIX}**")
    if criteria.strip() and strip_translate_suffix(criteria) != criteria.rstrip():
        st.caption("ℹ️ Prompt'a yazdığınız çeviri ifadesi zaten sabit ek olduğu için ikinci kez gönderilmeyecek.")
    full_criteria = with_translate_suffix(criteria) if criteria.strip() else ""

    c1, c2, c3 = st.columns(3)
    fetch_clicked = c1.button("1️⃣ Yorumları çek", width="stretch")
    filter_clicked = c2.button("2️⃣ Prompt'a göre ayıkla", type="primary", width="stretch")
    both_clicked = c3.button("⚡ Çek + ayıkla", width="stretch")

    links = [l.strip() for l in links_text.splitlines() if l.strip()]

    if fetch_clicked or both_clicked:
        if not links:
            st.warning("En az bir link girin.")
        else:
            state.fetched, state.results, state.errors = {}, None, []
            for link in links:
                with st.spinner(f"Çekiliyor: {link}"):
                    try:
                        if source == "YouTube":
                            state.fetched[link] = cached_youtube(link, int(max_comments), include_replies)
                        else:
                            state.fetched[link] = cached_eksi(link, int(max_pages), nice)
                    except Exception as e:
                        st.error(f"{link} çekilemedi: {e}")
            if state.fetched:
                state.run_id, state.run_filtered, state.run_source = history.new_run_id(), False, source
                save_current_run(full_criteria, model, backend)

    # ---------------- Maliyet tahmini ----------------
    fetched_items = [(link, items) for link, items in state.fetched.items() if items]
    if fetched_items and criteria.strip():
        requests = [msg for _, items in fetched_items for _, msg in build_requests(items, full_criteria, int(batch_size))]
        n_total = sum(len(items) for _, items in fetched_items)
        est_key = (model, criteria, int(batch_size), state.run_id)
        with st.expander("💰 Ayıklama öncesi maliyet tahmini", expanded=state.results is None):
            if backend == "api" and provider == "anthropic" and st.button("Token'ları API ile tam say (ücretsiz)"):
                try:
                    with st.spinner("Token sayılıyor..."):
                        state.exact_tokens = {est_key: count_input_tokens(model, requests, int(workers), api_key)}
                except Exception as e:
                    st.error(f"Token sayılamadı: {e}")
            est = estimate_cost(model, requests, n_total, state.exact_tokens.get(est_key))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("İstek (parti)", est["batches"], help=f"{n_total} öğe, {est['chars']:,} karakter")
            m2.metric("Giriş token", f"{'' if est['exact'] else '~'}{est['input_tokens']:,}")
            m3.metric("Çıkış token", "{:,}–{:,}".format(*est["output_tokens"]))
            m4.metric(f"Tahmini maliyet ({model_label})", "${:.2f}–${:.2f}".format(*est["cost"]))
            st.caption("🏷️ Bunun içinde gruplama özelliğinin payı: ${:.3f}–${:.3f}".format(*est["group_cost"]))
            notes = ["Giriş tokenı " + ("token sayma API'siyle ölçüldü." if est["exact"]
                                        else "karakter sayısından yaklaşık hesaplandı (~3 karakter/token).")]
            notes.append("Çıkış aralığı öğelerin %10–50'sinin seçileceği, seçilen her öğe için indeks/puan/grup "
                         "adı üretileceği ve (Haiku/Gemini Flash dışında) parti başına 300–3000 düşünme tokenı "
                         "varsayar.")
            if backend == "cli":
                notes.append("CLI backend'inde doğrudan ücret yok; bu tutar Claude aboneliğinin kullanım "
                             "kotasından düşer (CLI kendi sistem prompt'unu da ekler).")
            if provider != "anthropic":
                notes.append(f"{PROVIDERS[provider]['label']} fiyatlandırması tahminidir, güncel fiyatı sağlayıcıdan "
                             "teyit edin.")
            st.caption(" ".join(notes))

    if (filter_clicked or both_clicked) and state.fetched:
        if not criteria.strip():
            st.warning("Ayıklama prompt'u girin.")
        elif not key_ready:
            st.warning(f"{PROVIDERS[provider]['env']} tanımlı değil ve API anahtarı girilmedi.")
        else:
            results, errors = [], []
            bar = st.progress(0.0, text="Ayıklanıyor...")
            sources = [(link, items) for link, items in state.fetched.items() if items]
            for n, (link, items) in enumerate(sources):
                def progress(done, total, n=n):
                    bar.progress((n + done / total) / len(sources),
                                 text=f"Kaynak {n + 1}/{len(sources)} · parti {done}/{total}")
                picked, errs = filter_items(items, full_criteria, model, backend,
                                            int(batch_size), int(workers), progress, api_key)
                results += picked
                errors += [f"{link} → {e}" for e in errs]
            bar.empty()
            state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
            state.errors = errors
            if state.run_id is None or state.run_filtered:
                state.run_id = history.new_run_id()
            state.run_filtered = True
            save_current_run(full_criteria, model, backend)
    elif filter_clicked:
        st.warning("Önce yorumları çekin.")

    # ---------------- Sonuçlar ----------------
    all_items = [it for items in state.fetched.values() for it in items]
    if state.fetched:
        st.caption(" · ".join(f"**{(items[0]['title'] if items else link)[:60]}**: {len(items)}"
                              for link, items in state.fetched.items()) + f" — toplam {len(all_items)} öğe")

    for e in state.errors:
        st.error(e)

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
                    on_progress=lambda d, t: bar.progress(d / t, text=f"Çeviri partisi {d}/{t}"))
                bar.empty()
                state.errors = [e for e in state.errors if not e.startswith("Çeviri partisi")] + errs
                try:
                    old = history.load_run(state.run_id)[0]
                    save_current_run(old["criteria"], old["model"], old["backend"])
                except (OSError, TypeError):
                    st.warning("Kayıt bulunamadı; çeviri yalnızca bu oturumda geçerli.")
                st.rerun()
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
