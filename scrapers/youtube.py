"""YouTube yorumlarını yt-dlp ile çeker (API anahtarı gerekmez)."""
from datetime import datetime

import yt_dlp


def fetch_comments(url: str, max_comments: int = 500, include_replies: bool = True) -> list[dict]:
    # max_comments: [toplam, üst seviye, üst yorum başına yanıt, toplam yanıt]
    replies = "all" if include_replies else "0"
    opts = {
        "skip_download": True,
        "getcomments": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "max_comments": [str(max_comments), "all", replies, replies],
                "comment_sort": ["top"],
            }
        },
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_url = info.get("webpage_url") or url
    items = []
    for c in info.get("comments") or []:
        ts = c.get("timestamp")
        items.append({
            "source": "youtube",
            "title": info.get("title", ""),
            "url": video_url,
            "id": c.get("id"),
            "author": c.get("author", ""),
            "text": c.get("text", ""),
            "likes": c.get("like_count") or 0,
            "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "",
            "is_reply": c.get("parent") not in (None, "root"),
            "link": f"{video_url}&lc={c.get('id')}",
        })
    return items
