"""Web arayüzü: YouTube / Ekşi Sözlük'ten yorum çek, prompt'a göre Claude ile ayıkla."""
import pandas as pd
import streamlit as st

from llm_filter import MODELS, default_backend, filter_items
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


# ---------------- Kenar çubuğu ----------------
with st.sidebar:
    source = st.radio("Kaynak", ["YouTube", "Ekşi Sözlük"], horizontal=True)

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
    model_label = st.selectbox("Model", list(MODELS))
    batch_size = st.number_input("İstek başına yorum sayısı", 20, 500, 150, step=10)
    workers = st.number_input("Paralel istek", 1, 8, 4)

# ---------------- Ana alan ----------------
st.title("🔎 Yorum Ayıklayıcı")
placeholder = ("https://www.youtube.com/watch?v=...\nhttps://youtu.be/..." if source == "YouTube"
               else "https://eksisozluk.com/baslik-adi--123456\nveya doğrudan başlık adı")
links_text = st.text_area("Linkler (her satıra bir tane)", placeholder=placeholder, height=110)
criteria = st.text_area("Ayıklama prompt'u",
                        placeholder="Örn: Ürünü uzun süre kullanmış kişilerin somut deneyimleri ve "
                                    "yaşadıkları kronik sorunlar", height=90)

state = st.session_state
state.setdefault("fetched", {})     # link -> öğeler
state.setdefault("results", None)
state.setdefault("errors", [])

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
            picked, errs = filter_items(items, criteria, MODELS[model_label], backend,
                                        int(batch_size), int(workers), progress)
            results += picked
            errors += [f"{link} → {e}" for e in errs]
        bar.empty()
        state.results = sorted(results, key=lambda r: (-r["score"], -r["likes"]))
        state.errors = errors
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

if state.results is not None:
    st.subheader(f"✅ Seçilenler: {len(state.results)} / {len(all_items)}")
    cols = ["score", "text", "reason", "likes", "author", "date", "title", "link"]
    df = to_df(state.results, cols)
    st.dataframe(df, width="stretch", hide_index=True, column_config={
        **link_cfg, "score": st.column_config.ProgressColumn("puan", min_value=0, max_value=10, format="%d"),
        "reason": st.column_config.TextColumn("gerekçe", width="medium")})
    d1, d2 = st.columns(2)
    d1.download_button("CSV indir", df.to_csv(index=False).encode("utf-8-sig"), "secilenler.csv", "text/csv")
    d2.download_button("JSON indir", df.to_json(orient="records", force_ascii=False, indent=2),
                       "secilenler.json", "application/json")

if all_items:
    with st.expander(f"Tüm çekilen öğeler ({len(all_items)})"):
        cols = ["text", "likes", "author", "date", "is_reply", "title", "link"]
        raw = to_df(all_items, cols)
        st.dataframe(raw, width="stretch", hide_index=True, column_config=link_cfg)
        st.download_button("Ham veriyi CSV indir", raw.to_csv(index=False).encode("utf-8-sig"),
                           "tum_yorumlar.csv", "text/csv")
