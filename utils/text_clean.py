import html
import re

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Remove tags first, then decode entities."""
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    return html.unescape(text)