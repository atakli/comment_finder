"""Web arayüzü: YouTube / Ekşi Sözlük'ten yorum çek, prompt'a göre Claude ile ayıkla."""
import re

import pandas as pd
import streamlit as st

import history
from llm_filter import MODELS, build_requests, count_input_tokens, default_backend, estimate_cost, filter_items
from scrapers import fetch_comments, fetch_entries

st.set_page_config(page_title="Yorum Ayıklayıcı", page_icon="🔎", layout="wide")


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


def matches(item, query, fields=("text", "reason", "author", "title")):
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


def item_browser(rows, cols, column_config, key, query):
    """Tablo + tıklanan satırın tam metnini gösteren detay paneli."""
    table, detail = st.columns([3, 2])
    with table:
        event = st.dataframe(to_df(rows, cols), width="stretch", hide_index=True, column_config=column_config,
                             on_select="rerun", selection_mode="single-row", key=key)
    with detail, st.container(border=True):
        picked = event.selection.rows
        if not picked or picked[0] >= len(rows):
            st.caption("👈 Tam metni görmek için tablodan bir satıra tıklayın.")
            return
        it = rows[picked[0]]
        info = [f"**{md_escape(str(it.get('author') or '?'))}**", f"👍 {it.get('likes', 0)}",
                md_escape(str(it.get("date") or ""))]
        if it.get("is_reply"):
            info.append("↪️ yanıt")
        if it.get("score") is not None:
            info.append(f"⭐ {it['score']}/10")
        st.markdown(" · ".join(x for x in info if x))
        st.markdown(highlight(it.get("text"), query))
        if it.get("reason"):
            st.caption("Gerekçe")
            st.markdown(highlight(it["reason"], query))
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
    state.source, state.links_text, state.criteria = meta["source"], "\n".join(meta["links"]), meta["criteria"]
    label = next((k for k, v in MODELS.items() if v == meta["model"]), None)
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


# Yeni oturum (başka cihaz/sekme): URL'deki ?run=<id> kaydını, yoksa en son kaydı yükle.
# Kayıtlar sunucuda (data/runs/) olduğu için her cihaz aynı sonuçları görür.
if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = True
    run_ids = [r["id"] for r in history.list_runs()]
    wanted = st.query_params.get("run")
    if run_ids:
        load_run_into_state(wanted if wanted in run_ids else run_ids[0])


# ---------------- Kenar çubuğu ----------------
with st.sidebar:
    source = st.radio("Kaynak", ["YouTube", "Ekşi Sözlük"], horizontal=True, key="source")

    st.subheader("Çekme ayarları")
    if source == "YouTube":
        max_comments = st.number_input("Video başına en fazla yorum", 50, 20000, 1000, step=100)
        include_replies = st.checkbox("Yanıtları da dahil et", value=True)
    else:
        max_pages = st.number_input("Başlık başına en fazla sayfa (sayfa = 10 entry)", 1, 1000, 20)
        nice = st.checkbox("Şükela sıralaması (en çok favorilenenler önce)", value=True)

    st.subheader("Ayıklama (Claude)")
    backends = {"cli": "Claude Code CLI (abonelik)", "api": "Anthropic API (ANTHROPIC_API_KEY)"}
    backend = st.radio("Backend", list(backends), format_func=backends.get,
                       index=list(backends).index(default_backend()))
    model_label = st.selectbox("Model", list(MODELS), key="model_label")
    batch_size = st.number_input("İstek başına yorum sayısı", 20, 500, 150, step=10)
    workers = st.number_input("Paralel istek", 1, 8, 4)

# ---------------- Ana alan ----------------
st.title("🔎 Yorum Ayıklayıcı")
placeholder = ("https://www.youtube.com/watch?v=...\nhttps://youtu.be/..." if source == "YouTube"
               else "https://eksisozluk.com/baslik-adi--123456\nveya doğrudan başlık adı")
links_text = st.text_area("Linkler (her satıra bir tane)", placeholder=placeholder, height=110, key="links_text")
criteria = st.text_area("Ayıklama prompt'u",
                        placeholder="Örn: Ürünü uzun süre kullanmış kişilerin somut deneyimleri ve "
                                    "yaşadıkları kronik sorunlar", height=90, key="criteria")

state = st.session_state
state.setdefault("fetched", {})     # link -> öğeler
state.setdefault("results", None)
state.setdefault("errors", [])
state.setdefault("run_id", None)        # geçmişteki kayıt (data/runs/<id>)
state.setdefault("run_filtered", False)  # bu kayıtta ayıklama sonucu var mı
state.setdefault("run_source", source)
state.setdefault("exact_tokens", {})    # tahmin anahtarı -> count_tokens sonucu
state.setdefault("deleted_runs", [])    # bu oturumda silinenler (geri al yığını)
model = MODELS[model_label]

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
            save_current_run(criteria, model, backend)

# ---------------- Maliyet tahmini ----------------
fetched_items = [(link, items) for link, items in state.fetched.items() if items]
if fetched_items and criteria.strip():
    requests = [msg for _, items in fetched_items for _, msg in build_requests(items, criteria, int(batch_size))]
    n_total = sum(len(items) for _, items in fetched_items)
    est_key = (model, criteria, int(batch_size), state.run_id)
    with st.expander("💰 Ayıklama öncesi maliyet tahmini", expanded=state.results is None):
        if backend == "api" and st.button("Token'ları API ile tam say (ücretsiz)"):
            try:
                with st.spinner("Token sayılıyor..."):
                    state.exact_tokens = {est_key: count_input_tokens(model, requests, int(workers))}
            except Exception as e:
                st.error(f"Token sayılamadı: {e}")
        est = estimate_cost(model, requests, n_total, state.exact_tokens.get(est_key))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("İstek (parti)", est["batches"], help=f"{n_total} öğe, {est['chars']:,} karakter")
        m2.metric("Giriş token", f"{'' if est['exact'] else '~'}{est['input_tokens']:,}")
        m3.metric("Çıkış token", "{:,}–{:,}".format(*est["output_tokens"]))
        m4.metric(f"Tahmini maliyet ({model_label})", "${:.2f}–${:.2f}".format(*est["cost"]))
        notes = ["Giriş tokenı " + ("token sayma API'siyle ölçüldü." if est["exact"]
                                    else "karakter sayısından yaklaşık hesaplandı (~3 karakter/token).")]
        notes.append("Çıkış aralığı öğelerin %10–50'sinin seçileceği ve (Haiku dışında) parti başına "
                     "300–3000 düşünme tokenı varsayar.")
        if backend == "cli":
            notes.append("CLI backend'inde doğrudan ücret yok; bu tutar Claude aboneliğinin kullanım "
                         "kotasından düşer (CLI kendi sistem prompt'unu da ekler).")
        st.caption(" ".join(notes))

if (filter_clicked or both_clicked) and state.fetched:
    if not criteria.strip():
        st.warning("Ayıklama prompt'u girin.")
    else:
        results, errors = [], []
        bar = st.progress(0.0, text="Claude ayıklıyor...")
        sources = [(link, items) for link, items in state.fetched.items() if items]
        for n, (link, items) in enumerate(sources):
            def progress(done, total, n=n):
                bar.progress((n + done / total) / len(sources),
                             text=f"Kaynak {n + 1}/{len(sources)} · parti {done}/{total}")
            picked, errs = filter_items(items, criteria, model, backend,
                                        int(batch_size), int(workers), progress)
            results += picked
            errors += [f"{link} → {e}" for e in errs]
        bar.empty()
        state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
        state.errors = errors
        if state.run_id is None or state.run_filtered:  # önceki sonucu ezme, yeni kayıt aç
            state.run_id = history.new_run_id()
        state.run_filtered = True
        save_current_run(criteria, model, backend)
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
    query = st.text_input("🔍 Ara", placeholder="Metin, gerekçe, yazar veya başlıkta ara",
                          key="search_query").strip()
table_key = f"{state.run_id}_{fold(query)}"  # arama/kayıt değişince seçim sıfırlanır

if state.results is not None:
    shown = [r for r in state.results if matches(r, query)] if query else state.results
    st.subheader(f"✅ Seçilenler: {len(state.results)} / {len(all_items)}"
                 + (f" · aramada {len(shown)}" if query else ""))
    cols = ["score", "text", "reason", "likes", "author", "date", "title", "link"]
    item_browser(shown, cols, {
        **link_cfg, "score": st.column_config.ProgressColumn("puan", min_value=0, max_value=10, format="%d"),
        "reason": st.column_config.TextColumn("gerekçe", width="medium")}, f"res_{table_key}", query)
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

# ---------------- Geçmiş (kenar çubuğunun sonuna, bu çalıştırmadaki kayıtlar dahil) ----------------
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

# Açık kaydı URL'de tut: yenileyince/linki başka cihazda açınca aynı kayıt gelir
if state.run_id:
    st.query_params["run"] = state.run_id
else:
    st.query_params.pop("run", None)
