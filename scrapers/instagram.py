"""Instagram gönderi ve reel yorumlarını çeker (otomatik tarayıcı çerezi, manuel sessionid veya Apify API ile)."""
import os
import re
from datetime import datetime

from curl_cffi import requests

from .browser_cookies import format_cookie_header, get_instagram_cookies

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def extract_shortcode(url_or_code: str) -> str:
    """Instagram linkinden veya doğrudan verilen koddan shortcode'u çıkarır."""
    s = url_or_code.strip()
    m = re.search(r"/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", s):
        return s
    raise ValueError(f"Geçersiz Instagram linki veya shortcode: {url_or_code}")


def shortcode_to_media_id(shortcode: str) -> int:
    """Instagram shortcode'unu sayısal media_id'ye dönüştürür."""
    media_id = 0
    for char in shortcode:
        if char not in ALPHABET:
            raise ValueError(f"Geçersiz karakter shortcode içinde: {char}")
        media_id = media_id * 64 + ALPHABET.index(char)
    return media_id


def _fetch_via_apify(post_url: str, shortcode: str, max_comments: int, apify_token: str) -> list[dict]:
    """Apify Instagram Comment Scraper actor'ünü kullanarak yorumları çeker."""
    endpoint = "https://api.apify.com/v2/acts/apify~instagram-comment-scraper/run-sync-get-dataset-items"
    payload = {
        "directUrls": [post_url],
        "resultsLimit": max_comments,
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
        ts = c.get("timestamp") or ""
        date_str = ts[:10] if len(ts) >= 10 else ""
        items.append({
            "source": "instagram",
            "title": c.get("postTitle") or f"Instagram ({shortcode})",
            "url": post_url,
            "id": cid,
            "author": c.get("ownerUsername") or (c.get("user") or {}).get("username", ""),
            "text": c.get("text") or "",
            "likes": int(c.get("likesCount") or 0),
            "date": date_str,
            "is_reply": bool(c.get("isReply")),
            "link": f"{post_url}?comment_id={cid}" if cid else post_url,
        })
        if len(items) >= max_comments:
            break
    return items


def _fetch_via_cookie(
    post_url: str,
    shortcode: str,
    max_comments: int,
    include_replies: bool,
    sessionid: str | None = None,
    cookies_dict: dict | None = None,
) -> list[dict]:
    """Instagram dahili API'si üzerinden oturum çerezi ile yorumları çeker."""
    media_id = shortcode_to_media_id(shortcode)

    if cookies_dict:
        cookie_header = format_cookie_header(cookies_dict)
        csrftoken = cookies_dict.get("csrftoken")
    else:
        cookie_header = f"sessionid={(sessionid or '').strip()};"
        csrftoken = None

    headers = {
        "x-ig-app-id": "936619743392459",
        "x-asbd-id": "129477",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Cookie": cookie_header,
    }
    if csrftoken:
        headers["x-csrftoken"] = csrftoken

    session = requests.Session(impersonate="chrome")

    # Gönderi başlığını / açıklamasını almayı dene
    post_title = f"Instagram ({shortcode})"
    try:
        info_resp = session.get(f"https://www.instagram.com/api/v1/media/{media_id}/info/", headers=headers, timeout=20)
        if info_resp.status_code == 200:
            info_data = info_resp.json()
            caption = (info_data.get("items") or [{}])[0].get("caption", {})
            if caption and caption.get("text"):
                first_line = caption["text"].strip().splitlines()[0]
                post_title = first_line[:80] + ("..." if len(first_line) > 80 else "")
    except Exception:
        pass

    items = []
    min_id = None

    while len(items) < max_comments:
        params = {
            "can_support_threading": "true",
            "permalink_enabled": "false",
        }
        if min_id:
            params["min_id"] = min_id

        r = session.get(
            f"https://www.instagram.com/api/v1/media/{media_id}/comments/",
            headers=headers,
            params=params,
            timeout=30,
        )

        if r.status_code in (401, 403) or "login_required" in r.text or "checkpoint_required" in r.text:
            raise ValueError(
                "Instagram oturumu geçersiz veya süresi dolmuş. "
                "Lütfen tarayıcınızda instagram.com'a tekrar giriş yapın."
            )
        if r.status_code == 404:
            raise ValueError(f"Instagram gönderisi bulunamadı veya hesap gizli: {post_url}")
        r.raise_for_status()

        data = r.json()
        comments = data.get("comments") or []
        if not comments:
            break

        for c in comments:
            cid = str(c.get("pk") or "")
            ts = c.get("created_at")
            items.append({
                "source": "instagram",
                "title": post_title,
                "url": post_url,
                "id": cid,
                "author": (c.get("user") or {}).get("username", ""),
                "text": c.get("text") or "",
                "likes": int(c.get("comment_like_count") or 0),
                "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
                "is_reply": False,
                "link": f"{post_url}?comment_id={cid}",
            })
            if len(items) >= max_comments:
                break

            if include_replies:
                preview_replies = c.get("preview_child_comments") or c.get("child_comments") or []
                for reply in preview_replies:
                    rid = str(reply.get("pk") or "")
                    rts = reply.get("created_at")
                    items.append({
                        "source": "instagram",
                        "title": post_title,
                        "url": post_url,
                        "id": rid,
                        "author": (reply.get("user") or {}).get("username", ""),
                        "text": reply.get("text") or "",
                        "likes": int(reply.get("comment_like_count") or 0),
                        "date": datetime.fromtimestamp(rts).strftime("%Y-%m-%d") if rts else "",
                        "is_reply": True,
                        "link": f"{post_url}?comment_id={rid}",
                    })
                    if len(items) >= max_comments:
                        break

        if not data.get("has_more_comments") or not data.get("next_min_id"):
            break
        min_id = data["next_min_id"]

    return items


def fetch_instagram_comments(
    url: str,
    max_comments: int = 200,
    include_replies: bool = True,
    sessionid: str | None = None,
    apify_token: str | None = None,
) -> list[dict]:
    """Instagram gönderi veya reel linkinden yorumları çeker.

    Çerez verilmediyse kullanıcının açık tarayıcısındaki (Firefox, Chrome vb.)
    Instagram oturumunu otomatik olarak kullanır.
    """
    shortcode = extract_shortcode(url)
    post_url = f"https://www.instagram.com/p/{shortcode}/"

    sessionid = (sessionid or os.environ.get("INSTAGRAM_SESSIONID") or "").strip() or None
    apify_token = (apify_token or os.environ.get("APIFY_API_KEY") or "").strip() or None

    cookies_dict = None
    if not sessionid and not apify_token:
        # Otomatik olarak kullanıcının tarayıcısındaki çerezleri algıla
        cookies_dict, browser_name = get_instagram_cookies()
        if cookies_dict and "sessionid" in cookies_dict:
            sessionid = cookies_dict["sessionid"]

    if not sessionid and not apify_token:
        raise ValueError(
            "Instagram oturumu algılanamadı.\n\n"
            "Çözüm:\n"
            "Aynı bilgisayardaki tarayıcınızda (Firefox, Chrome, Brave vb.) instagram.com'a bir kez "
            "giriş yapmanız yeterlidir; sistem çerezi oradan otomatik olarak alacaktır."
        )

    if sessionid:
        return _fetch_via_cookie(
            post_url,
            shortcode,
            max_comments,
            include_replies,
            sessionid=sessionid,
            cookies_dict=cookies_dict,
        )
    return _fetch_via_apify(post_url, shortcode, max_comments, apify_token)
