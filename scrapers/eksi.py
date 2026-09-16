"""Ekşi Sözlük başlığındaki entry'leri çeker (curl_cffi ile Cloudflare'e takılmadan)."""
import time
from urllib.parse import quote, urlsplit

from bs4 import BeautifulSoup
from curl_cffi import requests

BASE = "https://eksisozluk.com"


def _topic_url(url_or_title: str) -> str:
    """Link verildiyse sorgu parametrelerini atar; başlık adı verildiyse ekşi aramasıyla başlığa gider."""
    s = url_or_title.strip()
    if s.startswith("http"):
        return BASE + urlsplit(s).path
    r = requests.get(f"{BASE}/?q={quote(s)}", impersonate="chrome", timeout=30)
    r.raise_for_status()
    return BASE + urlsplit(r.url).path


def _page(session, topic_url: str, page: int, nice: bool) -> BeautifulSoup:
    params = {"p": page}
    if nice:
        params["a"] = "nice"
    r = session.get(topic_url, params=params, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def fetch_entries(url_or_title: str, max_pages: int = 10, nice: bool = False, delay: float = 0.5) -> list[dict]:
    """nice=True: şükela (en çok favorilenen) sıralamasıyla çeker."""
    topic_url = _topic_url(url_or_title)
    session = requests.Session(impersonate="chrome")

    soup = _page(session, topic_url, 1, nice)
    title_el = soup.select_one("h1#title")
    if not title_el:
        raise ValueError(f"Başlık bulunamadı: {url_or_title}")
    title = title_el.get("data-title", "")
    pager = soup.select_one(".pager")
    page_count = int(pager["data-pagecount"]) if pager else 1

    items = []
    for page in range(1, min(page_count, max_pages) + 1):
        if page > 1:
            time.sleep(delay)
            soup = _page(session, topic_url, page, nice)
        for li in soup.select("#entry-item-list > li"):
            content = li.select_one(".content")
            if not content:
                continue
            for br in content.find_all("br"):
                br.replace_with("\n")
            date_el = li.select_one(".entry-date")
            items.append({
                "source": "eksi",
                "title": title,
                "url": topic_url,
                "id": li.get("data-id"),
                "author": li.get("data-author", ""),
                "text": content.get_text().strip(),
                "likes": int(li.get("data-favorite-count") or 0),
                "date": date_el.get_text(strip=True) if date_el else "",
                "is_reply": False,
                "link": f"{BASE}/entry/{li.get('data-id')}",
            })
    return items
