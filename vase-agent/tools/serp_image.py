# Serp image search (Google reverse image)
# Docs: https://serpapi.com/integrations/python
# https://serpapi.com/google-reverse-image

import os
from collections.abc import Mapping
from typing import Any

import dotenv
import serpapi

import serp_cache

dotenv.load_dotenv()

SERP_IMG_SEARCH_KEY = os.getenv("SERP_IMG_SEARCH_KEY")


def _json_endpoint(serp: Any) -> str:
    # SerpResults is a UserDict, not a plain dict; use Mapping
    if serp is None or isinstance(serp, (str, bytes)) or not isinstance(serp, Mapping):
        return ""
    meta = serp.get("search_metadata") or {}
    if not isinstance(meta, Mapping):
        return ""
    u = meta.get("json_endpoint")
    return str(u).strip() if u else ""


class ImageSearchTool:
    def __init__(self):
        self.api_key = SERP_IMG_SEARCH_KEY

    def run(self, image_urls: list[str]):
        """
        Agent tool:
        Reverse image search via SerpAPI google_reverse_image (up to 10 matches per image).
        Results are cached on disk (default vase-agent/tools/.serp_cache; override with SERP_CACHE_DIR).
        """
        if not self.api_key:
            return {
                "ok": False,
                "tool": "image_search",
                "error": "missing environment variable SERP_IMG_SEARCH_KEY",
                "results": [],
            }

        urls = [str(u).strip() for u in image_urls if str(u).strip()]
        if not urls:
            return {
                "ok": False,
                "tool": "image_search",
                "error": "image_urls is empty",
                "results": [],
            }

        try:
            client = serpapi.Client(api_key=self.api_key)
            blocks: list[dict[str, Any]] = []

            for image_url in urls:
                cpath = serp_cache.entry_path("google_reverse_image", image_url)
                cached = serp_cache.load_entry(cpath)
                if cached is not None:
                    blocks.append({**cached, "from_cache": True})
                    continue

                serp = client.search(
                    {
                        "engine": "google_reverse_image",
                        "image_url": image_url,
                    }
                )
                if isinstance(serp, str):
                    block = {
                        "query_image_url": image_url,
                        "json_endpoint": "",
                        "error": serp[:2000],
                        "visual_matches": [],
                    }
                    blocks.append(block)
                    continue

                err = serp.get("error")
                if err:
                    block = {
                        "query_image_url": image_url,
                        "json_endpoint": _json_endpoint(serp),
                        "error": str(err),
                        "visual_matches": [],
                    }
                    serp_cache.save_entry(cpath, block)
                    blocks.append(block)
                    continue

                raw_items = serp.get("image_results") or []
                matches: list[dict[str, Any]] = []
                for it in raw_items[:10]:
                    thumb = it.get("thumbnail") or it.get("favicon") or ""
                    if isinstance(thumb, dict):
                        thumb = thumb.get("link") or thumb.get("source") or ""
                    orig = it.get("original")
                    if not thumb and isinstance(orig, str):
                        thumb = orig
                    matches.append(
                        {
                            "rank": int(it.get("position") or len(matches) + 1),
                            "title": str(it.get("title") or ""),
                            "description": str(it.get("snippet") or ""),
                            "source_page_url": str(it.get("link") or ""),
                            "image_thumb_url": str(thumb),
                            "source": str(it.get("source") or ""),
                        }
                    )

                block = {
                    "query_image_url": image_url,
                    "json_endpoint": _json_endpoint(serp),
                    "visual_matches": matches,
                }
                # if not block.get("error"):
                #     serp_cache.save_entry(cpath, block)
                serp_cache.save_entry(cpath, block)
                blocks.append(block)

            return {
                "ok": any(b.get("visual_matches") for b in blocks),
                "tool": "image_search",
                "results": blocks,
            }

        except Exception as e:
            return {
                "ok": False,
                "tool": "image_search",
                "error": str(e),
                "results": [],
            }


if __name__ == "__main__":
    tool = ImageSearchTool()
    result = tool.run([
        "https://YOUR_IMAGE_HOST/images/B064BC80-2B28-4AC3-9B13-C128B5AF38E2_1_ac001001.jpg",
        "https://YOUR_IMAGE_HOST/images/4B339C11-B634-4F10-957C-300C4D462AE1_5_ac001001.jpg",
    ])
    import json
    print(json.dumps(result, indent=4))
    with open("img-eg-return.json", "w") as f:
        json.dump(result, f, indent=4)
    # print(tool.run([]))
