"""Facebook gönderi yorumlarını çeker (otomatik tarayıcı çerezi, Graph API veya Apify ile)."""
import os
import re
from datetime import datetime
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests

from .browser_cookies import format_cookie_header, get_facebook_cookies


def extract_facebook_info(url_or_id: str) -> tuple[str, str]:
    """Facebook linkinden veya ID'sinden post_id ve kanonik URL'yi çıkarır."""
    s = url_or_id.strip()
    if s.startswith("http"):
        post_url = s
        parsed = urlparse(s)
        qs = parse_qs(parsed.query)
        if "story_fbid" in qs:
            return qs["story_fbid"][0], post_url
        if "v" in qs:
            return qs["v"][0], post_url

        m = re.search(r"/(?:posts|reel|videos)/([A-Za-z0-9_]+)", parsed.path)
        if m:
            return m.group(1), post_url
        slug = parsed.path.strip("/").split("/")[-1]
        return slug or s, post_url
    return s, f"https://www.facebook.com/{s}"


def _fetch_via_apify(post_url: str, max_comments: int, apify_token: str) -> list[dict]:
    """Apify Facebook Comments Scraper actor'ünü kullanarak yorumları çeker."""
    endpoint = "https://api.apify.com/v2/acts/apify~facebook-comments-scraper/run-sync-get-dataset-items"
    payload = {
        "startUrls": [{"url": post_url}],
        "maxComments": max_comments,
    }
    r = requests.post(endpoint, params={"token": apify_token}, json=payload, timeout=120)
    if r.status_code in (401, 402):
        raise ValueError("Apify API anahtarı geçersiz veya yetersiz bakiye/kullanım hakkı.")
    r.raise_for_status()

    data = r.json()
    if not isinstance(data, list):
        return []

    items = []
    for c in data:
        cid = str(c.get("id") or "")
        ts = c.get("date") or c.get("timestamp") or ""
        date_str = str(ts)[:10] if len(str(ts)) >= 10 else ""
        items.append({
            "source": "facebook",
            "title": c.get("postTitle") or "Facebook Gönderisi",
            "url": post_url,
            "id": cid,
            "author": c.get("profileName") or c.get("author") or "",
            "text": c.get("text") or "",
            "likes": int(c.get("likesCount") or 0),
            "date": date_str,
            "is_reply": bool(c.get("isReply")),
            "link": c.get("commentUrl") or (f"{post_url}?comment_id={cid}" if cid else post_url),
        })
        if len(items) >= max_comments:
            break
    return items


def _fetch_via_graph_api(post_id: str, post_url: str, max_comments: int, access_token: str) -> list[dict]:
    """Facebook Graph API üzerinden yorumları çeker."""
    post_title = "Facebook Gönderisi"
    try:
        info_resp = requests.get(
            f"https://graph.facebook.com/v19.0/{post_id}",
            params={"fields": "message,story", "access_token": access_token},
            timeout=20,
        )
        if info_resp.status_code == 200:
            msg = info_resp.json().get("message") or info_resp.json().get("story") or ""
            if msg:
                first_line = msg.strip().splitlines()[0]
                post_title = first_line[:80] + ("..." if len(first_line) > 80 else "")
    except Exception:
        pass

    items = []
    next_url = f"https://graph.facebook.com/v19.0/{post_id}/comments"
    params = {
        "fields": "id,from,message,created_time,like_count,parent",
        "limit": min(max_comments, 100),
        "filter": "stream",
        "access_token": access_token,
    }

    while next_url and len(items) < max_comments:
        r = requests.get(next_url, params=params if next_url.startswith("https://graph.facebook.com/v19.0/") and params else None, timeout=30)
        if r.status_code in (400, 401, 403):
            err_msg = r.json().get("error", {}).get("message", "Erişim reddedildi.")
            raise ValueError(f"Facebook Graph API hatası: {err_msg}")
        r.raise_for_status()

        data = r.json()
        for c in data.get("data", []):
            cid = str(c.get("id") or "")
            created = c.get("created_time") or ""
            items.append({
                "source": "facebook",
                "title": post_title,
                "url": post_url,
                "id": cid,
                "author": (c.get("from") or {}).get("name", ""),
                "text": c.get("message") or "",
                "likes": int(c.get("like_count") or 0),
                "date": created[:10] if len(created) >= 10 else "",
                "is_reply": bool(c.get("parent")),
                "link": f"{post_url}?comment_id={cid}",
            })
            if len(items) >= max_comments:
                break

        params = None
        next_url = data.get("paging", {}).get("next")

    return items


def _fetch_via_cookies(post_url: str, post_id: str, max_comments: int, cookies: str) -> list[dict]:
    """Facebook web/mobil arayüzünden oturum çerezleri ile yorumları kazır."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        ),
        "Cookie": cookies.strip(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    session = requests.Session(impersonate="chrome")

    r = session.get(post_url, headers=headers, timeout=30)
    if "checkpoint" in r.url or "login" in r.url:
        raise ValueError(
            "Facebook oturum çerezleri geçersiz veya hesap doğrulaması gerekiyor. "
            "Lütfen tarayıcınızda facebook.com'a giriş yapın."
        )

    soup = BeautifulSoup(r.text, "html.parser")
    items = []
    post_title = "Facebook Gönderisi"
    h_title = soup.find("title")
    if h_title and h_title.get_text().strip():
        post_title = h_title.get_text().strip()

    # Yorum bloklarını ara
    comment_elements = soup.select("div[data-commentid], div[id^='comment_'], div[data-sigil*='comment']")
    for el in comment_elements:
        cid = el.get("data-commentid") or el.get("id") or ""
        author_el = el.select_one("h3, strong, a[href*='profile'], .actor")
        author = author_el.get_text().strip() if author_el else ""
        text_el = el.select_one("[data-comment-previews], div[dir='auto'], span[dir='auto']")
        text = text_el.get_text().strip() if text_el else el.get_text().strip()
        if not text or (author and text == author):
            continue

        items.append({
            "source": "facebook",
            "title": post_title,
            "url": post_url,
            "id": str(cid),
            "author": author,
            "text": text,
            "likes": 0,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "is_reply": False,
            "link": f"{post_url}?comment_id={cid}" if cid else post_url,
        })
        if len(items) >= max_comments:
            break

    if not items:
        # Facebook web dinamik JS korumaları nedeniyle doğrudan HTML ayrıştırma bazen kısıtlanır
        raise ValueError(
            "Facebook bu gönderideki yorumları dinamik JavaScript ile koruyor. "
            "Gönderi yorumlarını güvenle çekmek için kenar çubuğundaki Gelişmiş Ayarlar'dan "
            "Apify API Token veya Facebook Graph API Access Token kullanabilirsiniz."
        )

    return items


def fetch_facebook_comments(
    url: str,
    max_comments: int = 200,
    cookies: str | None = None,
    access_token: str | None = None,
    apify_token: str | None = None,
) -> list[dict]:
    """Facebook gönderi veya video linkinden yorumları çeker.

    Çerez verilmediyse kullanıcının açık tarayıcısındaki (Firefox, Chrome vb.)
    Facebook oturumunu otomatik olarak kullanır.
    """
    post_id, post_url = extract_facebook_info(url)

    cookies = (cookies or os.environ.get("FACEBOOK_COOKIES") or "").strip() or None
    access_token = (access_token or os.environ.get("FACEBOOK_ACCESS_TOKEN") or "").strip() or None
    apify_token = (apify_token or os.environ.get("APIFY_API_KEY") or "").strip() or None

    if not cookies and not access_token and not apify_token:
        # Tarayıcıdan otomatik algıla
        auto_cookies, browser_name = get_facebook_cookies()
        if auto_cookies:
            cookies = format_cookie_header(auto_cookies)

    if not cookies and not access_token and not apify_token:
        raise ValueError(
            "Facebook oturumu algılanamadı.\n\n"
            "Çözüm:\n"
            "Aynı bilgisayardaki tarayıcınızda (Firefox, Chrome, Brave vb.) facebook.com'a bir kez "
            "giriş yapmanız yeterlidir; sistem oturumu oradan otomatik olarak alacaktır. "
            "Dilerseniz kenar çubuğundan Apify API anahtarı veya Graph API erişim jetonu da girebilirsiniz."
        )

    if apify_token:
        return _fetch_via_apify(post_url, max_comments, apify_token)
    if access_token:
        return _fetch_via_graph_api(post_id, post_url, max_comments, access_token)
    return _fetch_via_cookies(post_url, post_id, max_comments, cookies)
