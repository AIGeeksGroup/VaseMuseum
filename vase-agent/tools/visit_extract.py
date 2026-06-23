import os
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from readability import Document


def _env_ms(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
        return v if v >= 100 else default
    except ValueError:
        return default


def _env_sec(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return default


class VisitTool:
    """
    Playwright visit + Readability.

    Durations (override via env):
      VASE_VISIT_GOTO_TIMEOUT_MS — page.goto max wait (default 60000)
      VASE_VISIT_POST_LOAD_SLEEP_SEC — sleep after load for SPA/late JS (default 2.0)
      VASE_VISIT_WAIT_UNTIL — load | domcontentloaded | commit | networkidle (default networkidle)
    """

    def run(self, url: str, goal: str = ""):
        timeout_ms = _env_ms("VASE_VISIT_GOTO_TIMEOUT_MS", 60000)
        post_sleep = _env_sec("VASE_VISIT_POST_LOAD_SLEEP_SEC", 2.0)
        wait_until_raw = (os.getenv("VASE_VISIT_WAIT_UNTIL") or "networkidle").strip().lower()
        allowed = ("load", "domcontentloaded", "commit", "networkidle")
        wait_until = wait_until_raw if wait_until_raw in allowed else "networkidle"

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                page.goto(url, wait_until=wait_until, timeout=timeout_ms)

                time.sleep(post_sleep)

                # page.wait_for_selector("body")

                html = page.content()

                browser.close()

            soup = BeautifulSoup(html, "html.parser")
            full_text = soup.get_text(separator="\n").strip()

            doc = Document(html)
            readable_html = doc.summary()
            readable_text = BeautifulSoup(readable_html, "html.parser").get_text("\n").strip()

            return {
                "url": url,
                "goal": goal,
                "title": doc.title(),
                "html": html[:50000],
                "text": full_text[:20000],
                "readable": readable_text[:10000],
            }

        except Exception as e:
            return {
                "url": url,
                "goal": goal,
                "error": str(e),
                "title": "",
                "html": "",
                "text": "",
                "readable": ""
            }

if __name__ == "__main__":
    tool = VisitTool()
    # print(tool.run("https://www .metmuseum.org/art/collection/search/251494", "extract the content of the page"))
    # print(tool.run("https://en.wikipedia.org/wiki/Pottery_of_ancient_Greece", "extract the content of the page"))
    import json
    results = []
    # result = tool.run("http://ark.dasch.swiss/ark:/72163/080e-73b1c12daba93-2", "extract the content of the page")
    # results.append(result)
    # print(json.dumps(result, indent=4))
    # result = tool.run("http://www.beazley.ox.ac.uk/record/64B75012-4B36-4DB9-9E0F-D8D6A49214FF")
    # results.append(result)
    # print(json.dumps(result, indent=4))
    result = tool.run("https://www.britishmuseum.org/collection/object/G_1864-1007-1697", "extract the content of the page")
    results.append(result)
    print(json.dumps(result, indent=4))
    with open("visit_extract_2.json", "w") as f:
        json.dump(results, f, indent=4)