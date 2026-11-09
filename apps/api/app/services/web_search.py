"""Fast live web/news grounding for current-events, politics, and external factual questions.

Only runs when `settings.allow_external_services` is True and the user's question
is not small talk or a purely personal family-memory query.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date

import httpx
from loguru import logger

from app.core.config import settings

_CURRENT_EVENTS_PATTERNS = re.compile(
    r"(?:"
    r"חדשות|אקטואליה|פוליטיק|ממשלה|כנסת|בחירות|ראש הממשלה|שר ה|בג\"ץ|קואליציה|אופוזיציה|"
    r"מלחמ|ביטחון|צה\"ל|חיל האוויר|איראן|לבנון|עזה|ארה\"ב|נשיא|טראמפ|ביידן|אירופה|רוסיה|אוקראינה|סין|"
    r"כלכל|מדד|ריבית|דולר|שקל|יורו|בורסה|מניו?ת|אינפלציה|תקציב|מחיר|"
    r"היום ב|השבוע ב|לאחרונה|עכשיו ב|מה קורה ב|מה המצב ב|בעולם|בישראל|"
    r"איזה דגם|מפרט|טכנולוגי|בינה מלאכותית|סייבר|פישינג|חומרה|מעבד|כרטיס מסך"
    r")",
    re.IGNORECASE,
)

_GENERAL_NEWS_TRIGGERS = re.compile(
    r"(?:חדשות|אקטואליה|פוליטיק|מה קורה עכשיו|מה המצב בארץ|מה המצב בעולם|כותרות)",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset({
    "מה", "איך", "למה", "כמה", "איזה", "אתה", "חושב", "אומר", "יודע", "על", "של",
    "עם", "את", "זה", "זאת", "היום", "עכשיו", "כרגע", "בכלל", "לי", "לך", "יש",
})

_TAG_RE = re.compile(r"<[^>]+>")


def needs_web_grounding(query: str) -> bool:
    """Return True if `query` asks about politics, news, world affairs, markets, or tech/specs."""
    return bool(_CURRENT_EVENTS_PATTERNS.search(query))


def _clean_search_terms(query: str) -> str:
    words = [w for w in re.sub(r"[^\w\s\"'-]", " ", query).split() if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(words[:6]) or query.strip()


_WEATHER_OR_GOSSIP = re.compile(
    r"(?:תחזית|מזג האוויר|גשם|הצפות|סופות רעמים|סלבס|ריאליטי|האח הגדול|אינסטגרם|טיקטוק)",
    re.IGNORECASE,
)


def _parse_rss_items(xml_text: str, limit: int) -> list[str]:
    root = ET.fromstring(xml_text)
    items: list[str] = []
    for item in root.findall(".//item"):
        title = html.unescape(_TAG_RE.sub("", item.findtext("title") or "")).strip()
        desc = html.unescape(_TAG_RE.sub("", item.findtext("description") or "")).strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title or _WEATHER_OR_GOSSIP.search(title):
            continue
        snippet = f"{title} — {desc[:110]}" if desc and desc not in title else title
        items.append(f"- {snippet} ({pub[:16]})" if pub else f"- {snippet}")
        if len(items) >= limit:
            break
    return items


def fetch_web_context(query: str, max_items: int = 5) -> list[str]:
    """Fetch fresh Hebrew news/web headlines for `query` (~0.5s). Returns [] on any error."""
    if not settings.allow_external_services or not needs_web_grounding(query):
        return []

    headlines: list[str] = []
    seen: set[str] = set()

    def add_items(items: list[str]) -> None:
        for item in items:
            key = item[:45]
            if key not in seen and len(headlines) < max_items:
                seen.add(key)
                headlines.append(item)

    with httpx.Client(timeout=3.5, follow_redirects=True) as client:
        if _GENERAL_NEWS_TRIGGERS.search(query):
            for rss_url in (
                "https://www.ynet.co.il/Integration/StoryRss1854.xml",
                "https://rss.walla.co.il/feed/1",
            ):
                try:
                    res = client.get(rss_url)
                    if res.status_code == 200:
                        add_items(_parse_rss_items(res.text, 3))
                except Exception as e:
                    logger.debug("RSS {} skipped: {}", rss_url, e)

        cleaned_q = _clean_search_terms(query)
        gnews_url = (
            f"https://news.google.com/rss/search?"
            f"q={urllib.parse.quote(cleaned_q)}&hl=he&gl=IL&ceid=IL:he"
        )
        try:
            res = client.get(gnews_url)
            if res.status_code == 200:
                add_items(_parse_rss_items(res.text, max_items))
        except Exception as e:
            logger.debug("Google News RSS skipped: {}", e)

    logger.info("Web grounding fetched {} items for query ({} chars)", len(headlines), len(query))
    return headlines


def format_web_context(headlines: list[str]) -> str:
    if not headlines:
        return ""
    today = date.today().isoformat()
    return (
        f"מבזקים וכותרות חיות מהרשת הישראלית והעולמית (נכון להיום, {today} — מידע מאומת ועדכני):\n"
        + "\n".join(headlines)
        + "\nהתבסס ישירות על הדיווחים האלה כדי לתת תשובה מעודכנת, חדה ודעתנית בסגנון האישי שבפרופיל (אל תגיד שאין לך גישה לחדשות עדכניות)."
    )
