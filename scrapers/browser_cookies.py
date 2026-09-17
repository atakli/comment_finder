"""Kullanıcının sistemindeki tarayıcılardan (Firefox, Chrome, Brave, Edge vb.) oturum çerezlerini otomatik okur."""
import logging
from yt_dlp.cookies import extract_cookies_from_browser

logger = logging.getLogger(__name__)

SUPPORTED_BROWSERS = ("firefox", "chrome", "brave", "chromium", "edge", "opera", "vivaldi")


def get_instagram_cookies(preferred_browser: str | None = None) -> tuple[dict | None, str | None]:
    """Instagram oturum çerezlerini (özellikle sessionid) tarayıcıdan otomatik çıkarır.

    Dönüş: (cookies_dict, browser_name) veya (None, None)
    """
    browsers = [preferred_browser] if preferred_browser else SUPPORTED_BROWSERS
    for b in browsers:
        if not b:
            continue
        try:
            jar = extract_cookies_from_browser(b)
            ig_cookies = {c.name: c.value for c in jar if "instagram" in (c.domain or "")}
            if "sessionid" in ig_cookies:
                return ig_cookies, b
        except Exception:
            continue
    return None, None


def get_facebook_cookies(preferred_browser: str | None = None) -> tuple[dict | None, str | None]:
    """Facebook oturum çerezlerini (c_user, xs vb.) tarayıcıdan otomatik çıkarır.

    Dönüş: (cookies_dict, browser_name) veya (None, None)
    """
    browsers = [preferred_browser] if preferred_browser else SUPPORTED_BROWSERS
    for b in browsers:
        if not b:
            continue
        try:
            jar = extract_cookies_from_browser(b)
            fb_cookies = {c.name: c.value for c in jar if "facebook" in (c.domain or "")}
            if "c_user" in fb_cookies and "xs" in fb_cookies:
                return fb_cookies, b
        except Exception:
            continue
    return None, None


def format_cookie_header(cookie_dict: dict) -> str:
    """Çerez sözlüğünü HTTP Cookie başlığı biçimine dönüştürür."""
    return "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
