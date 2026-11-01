"""
Trend Discovery Module ("What People Look Into") for MoneyPrinterTurbo.

Fetches live trending queries and headlines from:
1. Google Autocomplete API (Free & Fast)
2. Reddit JSON Feeds (No API key required: r/technology, r/science, r/Damnthatsinteresting)
3. Hacker News API (Top stories for cutting-edge tech breakthroughs)

Then injects `{{TRENDING_TOPICS_ARRAY}}` into the LLM prompt so the LLM picks the
cleanest, most viral Halal-friendly topic(s), discards anything inappropriate,
and optionally generates a complete viral video package (subject, narration script,
safe 3D visual scene prompts, and SFX cues).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote_plus

import requests
from loguru import logger

from app.services import llm

GOOGLE_AUTOCOMPLETE_URL = "http://suggestqueries.google.com/complete/search"
DEFAULT_GOOGLE_PREFIXES = (
    "how does ",
    "why does science ",
    "new technology ",
    "mind blowing fact about ",
    "what happens to your body when ",
    "how do doctors ",
)

REDDIT_FEEDS = {
    "r/technology": "https://www.reddit.com/r/technology/top/.json?t=day&limit=10",
    "r/science": "https://www.reddit.com/r/science/top/.json?t=day&limit=10",
    "r/Damnthatsinteresting": "https://www.reddit.com/r/Damnthatsinteresting/top/.json?t=day&limit=10",
}

HACKER_NEWS_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HACKER_NEWS_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MoneyPrinterTurbo-TrendDiscovery/1.0"

VIRAL_TREND_SELECTION_PROMPT_TEMPLATE = """
# Role: Viral Halal-Friendly 3D Science & Tech Producer (Zack D. Films Style)

## Live Trending Data (What People Are Searching & Reading Right Now):
{{TRENDING_TOPICS_ARRAY}}

## Your Task:
Analyze the raw live trends above and select/adapt the TOP {count} cleanest, most viral, curiosity-inducing video topics that work as a 3D cutaway explainer animation (like Zack D. Films).

## Strict Halal & Content Safety Filter:
1. DISCARD any topic involving politics, wars, celebrities, music, alcohol, pork, gambling, dating/romance, nudity/immodesty, religion/idols, or controversy.
2. KEEP & ADAPT pure science, human anatomy, medical procedures, physics, survival mechanisms, engineering, space, or cutting-edge technology breakthroughs.
3. Rewrite each selected trend into a punchy 6-14 word viral hook (e.g., "What Happens Inside Your Body When...", "How New Solid-State Batteries Actually Work", "Why Scientists Just Discovered...").
4. {language_instruction}

## Output Format:
Return ONLY a valid JSON array of strings (no markdown, no explanation):
["Viral Topic 1", "Viral Topic 2"]
""".strip()

VIRAL_VIDEO_BLUEPRINT_PROMPT_TEMPLATE = """
# Role: Viral 3D Short-Form Video Director & Scriptwriter (Zack D. Films Style)

## Live Trending Data (What People Are Searching & Reading Right Now):
{{TRENDING_TOPICS_ARRAY}}

## Your Task:
1. Pick the SINGLE cleanest, most viral, 100% Halal-friendly topic from `{{TRENDING_TOPICS_ARRAY}}` above (discarding anything political, inappropriate, romantic, or non-Halal).
2. Turn it into a complete viral short-form video package with:
   - `topic`: A punchy 6-14 word viral title/hook.
   - `source_trend`: The original raw trend you picked from the list.
   - `script`: A 1-paragraph spoken narration script (85-115 words) in the exact conversational style of Zack D. Films. Start immediately with "When you...", "If you...", or "The second you...". Use plain, everyday English (zero poetic metaphors) walking step-by-step through the physical, anatomical, or technological cause-and-effect process.
   - `scene_prompts`: Exactly 10 chronological, safe 3D visual prompts (8-14 words each, no commas inside a prompt) using Zack D. Films 3D shot types on a bright sky-blue background or bright room lighting.
   - `sfx_cues`: 4-6 concise sound effect cues matching the scenes (e.g., "subtle whoosh transition", "low mechanical hum", "crisp pop sound").
3. {language_instruction}

## Output Format:
Return ONLY a valid JSON object with keys `"topic"`, `"source_trend"`, `"script"`, `"scene_prompts"`, and `"sfx_cues"`.
""".strip()


VIRAL_ZACK_D_PROMPT = """
### ROLE
You are the lead scriptwriter and 3D visual director for a viral short-form channel like "Zack D. Films" and "Veritasium".
Your specialty is taking complex Science, Technology, and Physics trends and explaining them to a 12-year-old using visceral, physical analogies, high-retention hooks, and strict Halal compliance.

### INPUT TREND DATA
{trending_topics}

### STRICT HALAL GUARDRAILS (MANDATORY)
- ZERO music references, gambling, alcohol, dating, astrology, or inappropriate anatomy.
- Visuals must strictly focus on mechanical objects, electronics, physics simulations, nature, space, or microscopic/anatomical cross-sections without nudity or full human bodies.

### VIRAL SCRIPT FORMULA (30-40 SECONDS, ~85-100 WORDS)
1. THE 3-SECOND HOOK (Sentence 1): NEVER start with "When you..." or "Today we will see...". Start with a shocking physical reality, a counter-intuitive fact, or "If you sliced open a [object]...".
2. PHYSICAL ANALOGIES (Body): Never use dry textbook words like "bias" or "vectors" without comparing them to something physical (e.g., "billions of microscopic switches", "a maze of glowing glass tunnels"). Every sentence must trigger a clear visual image in the viewer's brain.
3. PACING: Short, punchy sentences (max 12 words per sentence). Zero fluff.
4. LOOPING ENDING (Final sentence): End with a mind-blowing conclusion that seamlessly loops back to the first sentence.

### VISUAL STRATEGY (CRITICAL TO AVOID BORING ABSTRACT BACKGROUNDS)
You must provide TWO types of visual queries for every scene:
1. `cgi_3d_prompt`: For AI Image/Video Generators (Flux / Midjourney / Kling). Must specify: "3D cutaway cross-section animation in the style of Zack D Films, glossy plastic and metal textures, studio rim lighting, dark background, macro close-up of [exact physical mechanism], Unreal Engine 5 render, 9:16 vertical".
2. `stock_search_keyword`: For Pexels/Pixabay fallback. NEVER use abstract words like "math nodes", "vector flow", or "error graph" (these return ugly psychedelic screensavers). ONLY use 2-3 words describing REAL, TANGIBLE PHYSICAL OBJECTS filmed in close-up (e.g., "silicon wafer macro", "fiber optic cables", "circuit board soldering", "data center server", "electric spark slow motion").
Provide 8 to 10 `cgi_3d_prompts` (one per scene, in the chronological order of the narration) and 8 `video_keywords`.

### OUTPUT FORMAT (STRICT JSON ONLY)
Return ONLY valid JSON with no markdown formatting:
{
  "selected_topic": "Viral clickable title (e.g., What Actually Happens Inside AI Chips?)",
  "viral_hook_type": "Curiosity Gap / Cutaway Mechanism",
  "source_trend": "The raw trend line you picked from the input",
  "script": "The complete spoken narration text (85-100 words, fast-paced, Zack D. Films style).",
  "cgi_3d_prompts": [
    "3D cutaway cross-section in the style of Zack D Films, macro view inside a glowing silicon microchip, millions of tiny golden logic gates flipping on and off, dark background, Unreal Engine 5, 9:16"
  ],
  "video_keywords": [
    "microchip macro",
    "fiber optic light",
    "circuit board closeup",
    "server room cables",
    "laser beam glass",
    "computer processor",
    "robotic arm factory",
    "electric arc macro"
  ],
  "sfx_cues": [
    "crisp mechanical click",
    "high-tech electrical hum",
    "deep sub-bass impact"
  ]
}
""".strip()


def fetch_google_autocomplete_trends(
    prefixes: tuple[str, ...] | list[str] = DEFAULT_GOOGLE_PREFIXES,
    limit_per_prefix: int = 3,
    timeout: float = 6.0,
) -> list[dict[str, str]]:
    """Query Google's free Firefox autocomplete endpoint for live search queries."""
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for prefix in prefixes:
        try:
            url = f"{GOOGLE_AUTOCOMPLETE_URL}?client=firefox&q={quote_plus(prefix)}"
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
                count = 0
                for suggestion in payload[1]:
                    text = str(suggestion or "").strip()
                    if not text or len(text) < 10:
                        continue
                    norm = text.lower()
                    if norm in seen:
                        continue
                    seen.add(norm)
                    results.append({"title": text, "source": f"Google ({prefix.strip()})"})
                    count += 1
                    if count >= limit_per_prefix:
                        break
        except Exception as exc:
            logger.debug(f"Google autocomplete fetch skipped for '{prefix}': {exc}")

    return results


def fetch_reddit_trends(
    subreddits: dict[str, str] | None = None,
    limit_per_sub: int = 5,
    timeout: float = 8.0,
) -> list[dict[str, str]]:
    """Fetch top daily posts from science and tech Reddit JSON feeds."""
    feeds = subreddits or REDDIT_FEEDS
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for sub_name, url in feeds.items():
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            children = (data.get("data") or {}).get("children") or []
            count = 0
            for child in children:
                post = (child or {}).get("data") or {}
                if post.get("over_18") or post.get("stickied"):
                    continue
                title = str(post.get("title") or "").strip()
                if not title or len(title) < 12:
                    continue
                norm = title.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                results.append({"title": title, "source": f"Reddit {sub_name}"})
                count += 1
                if count >= limit_per_sub:
                    break
        except Exception as exc:
            logger.debug(f"Reddit JSON feed skipped for {sub_name}: {exc}")

    return results


def fetch_hacker_news_trends(
    limit: int = 8,
    timeout: float = 6.0,
) -> list[dict[str, str]]:
    """Pull top stories from Hacker News Firebase API."""
    results: list[dict[str, str]] = []
    try:
        resp = requests.get(
            HACKER_NEWS_TOP_URL,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return results
        story_ids = resp.json()
        if not isinstance(story_ids, list):
            return results

        candidate_ids = story_ids[: max(limit * 2, 12)]

        def _fetch_item(item_id: int) -> dict[str, Any] | None:
            try:
                r = requests.get(
                    HACKER_NEWS_ITEM_URL.format(item_id=item_id),
                    headers={"User-Agent": _USER_AGENT},
                    timeout=timeout,
                )
                if r.status_code == 200:
                    return r.json()
            except Exception:
                return None
            return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_item, sid): sid for sid in candidate_ids}
            for future in as_completed(futures):
                item = future.result()
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                if not title or title.lower().startswith(("show hn:", "ask hn:", "tell hn:")):
                    continue
                results.append({"title": title, "source": "Hacker News"})
                if len(results) >= limit:
                    break
    except Exception as exc:
        logger.debug(f"Hacker News fetch skipped: {exc}")

    return results[:limit]


def fetch_live_trending_topics(
    sources: list[str] | None = None,
    max_total: int = 18,
    custom_query_prefix: str = "",
) -> list[dict[str, str]]:
    """
    Fetch 10-18 live raw headlines/queries from Google Autocomplete, Reddit JSON,
    and Hacker News concurrently.
    """
    enabled_sources = set(sources or ["google", "reddit", "hackernews"])
    combined: list[dict[str, str]] = []

    prefixes = list(DEFAULT_GOOGLE_PREFIXES)
    if custom_query_prefix and custom_query_prefix.strip():
        prefixes.insert(0, custom_query_prefix.strip() + " ")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        if "google" in enabled_sources:
            futures[executor.submit(fetch_google_autocomplete_trends, prefixes, 3)] = "google"
        if "reddit" in enabled_sources:
            futures[executor.submit(fetch_reddit_trends, None, 4)] = "reddit"
        if "hackernews" in enabled_sources:
            futures[executor.submit(fetch_hacker_news_trends, 6)] = "hackernews"

        for future in as_completed(futures):
            try:
                items = future.result() or []
                combined.extend(items)
            except Exception as exc:
                logger.warning(f"Trend source failed: {exc}")

    # Deduplicate while preserving balanced interleaving across sources
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in combined:
        title = str(item.get("title") or "").strip()
        key = re.sub(r"\W+", " ", title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_total:
            break

    return deduped


def format_trending_topics_array(raw_trends: list[dict[str, str]] | list[str]) -> str:
    """Format raw trend items as a JSON array string to replace {{TRENDING_TOPICS_ARRAY}}."""
    formatted: list[str] = []
    for item in raw_trends:
        if isinstance(item, dict):
            src = item.get("source", "Live Trend")
            title = item.get("title", "")
            if title:
                formatted.append(f"[{src}] {title}")
        elif isinstance(item, str) and item.strip():
            formatted.append(item.strip())
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def generate_viral_topics_from_trends(
    sources: list[str] | None = None,
    count: int = 5,
    language: str = "",
    custom_query_prefix: str = "",
    app_config=None,
) -> dict[str, Any]:
    """
    Workflow:
    1. Fetch 10-15 raw headlines/queries from Google Autocomplete, Reddit JSON, and Hacker News.
    2. Inject into {{TRENDING_TOPICS_ARRAY}} in the prompt.
    3. Ask LLM to filter for Halal-friendly, high-retention viral topics.
    """
    raw_trends = fetch_live_trending_topics(
        sources=sources,
        max_total=16,
        custom_query_prefix=custom_query_prefix,
    )
    trending_array_str = format_trending_topics_array(raw_trends)

    language_instruction = (
        f"Write the output topics in language '{language}'."
        if (language or "").strip()
        else "Write the output topics in English."
    )

    prompt = (
        VIRAL_TREND_SELECTION_PROMPT_TEMPLATE.replace(
            "{{TRENDING_TOPICS_ARRAY}}", trending_array_str
        )
        .replace("{count}", str(max(1, min(int(count or 5), 10))))
        .replace("{language_instruction}", language_instruction)
    )

    curated_topics: list[str] = []
    try:
        response = (
            llm._generate_response(prompt)
            if app_config is None
            else llm._generate_response(prompt, app_config=app_config)
        )
        if isinstance(response, str) and not response.startswith("Error: "):
            cleaned = llm._strip_code_fence(response)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = llm._extract_embedded_json(cleaned, "[")
            if isinstance(parsed, list):
                curated_topics = [str(x).strip() for x in parsed if str(x).strip()]
    except Exception as exc:
        logger.warning(f"LLM trend curation failed, falling back to raw trends: {exc}")

    if not curated_topics:
        curated_topics = [item["title"] for item in raw_trends[:count] if item.get("title")]

    return {
        "raw_trends": raw_trends,
        "curated_topics": curated_topics[:count],
        "trending_array_json": trending_array_str,
    }


def generate_viral_video_blueprint(
    sources: list[str] | None = None,
    language: str = "",
    custom_query_prefix: str = "",
    raw_trends_override: list[dict[str, str]] | None = None,
    app_config=None,
) -> dict[str, Any]:
    """
    Full 1-Click Viral Video Generator from Live Trends:
    - Fetches live trends (Google Autocomplete + Reddit + Hacker News)
    - Injects `{{TRENDING_TOPICS_ARRAY}}`
    - LLM picks the cleanest, most viral Halal-friendly topic and outputs:
      topic, script, 10 safe 3D visual scene prompts, and SFX cues.
    """
    raw_trends = (
        raw_trends_override
        if raw_trends_override is not None
        else fetch_live_trending_topics(
            sources=sources,
            max_total=15,
            custom_query_prefix=custom_query_prefix,
        )
    )
    trending_array_str = format_trending_topics_array(raw_trends)
    language_note = (
        f"\n### LANGUAGE\nWrite `selected_topic` and `script` in language '{language}'. "
        "Keep `cgi_3d_prompts` and `video_keywords` in English.\n"
        if (language or "").strip() and not str(language).lower().startswith("en")
        else ""
    )

    # str.replace (not .format) because the prompt contains literal JSON braces.
    prompt = VIRAL_ZACK_D_PROMPT.replace("{trending_topics}", trending_array_str) + language_note

    try:
        response = (
            llm._generate_response(prompt)
            if app_config is None
            else llm._generate_response(prompt, app_config=app_config)
        )
        if isinstance(response, str) and not response.startswith("Error: "):
            cleaned = llm._strip_code_fence(response)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = llm._extract_embedded_json(cleaned, "{")
            topic = (
                (parsed.get("selected_topic") or parsed.get("topic"))
                if isinstance(parsed, dict)
                else ""
            )
            if isinstance(parsed, dict) and topic and parsed.get("script"):
                def _clean_list(key: str) -> list[str]:
                    return [
                        re.sub(r"\s+", " ", str(s)).strip()
                        for s in (parsed.get(key) or [])
                        if str(s).strip()
                    ]

                cgi_prompts = _clean_list("cgi_3d_prompts") or _clean_list("scene_prompts")
                return {
                    "success": True,
                    "topic": str(topic).strip(),
                    "viral_hook_type": str(parsed.get("viral_hook_type", "")).strip(),
                    "source_trend": str(parsed.get("source_trend", "")).strip(),
                    "script": str(parsed.get("script", "")).strip(),
                    "scene_prompts": cgi_prompts,
                    "cgi_3d_prompts": cgi_prompts,
                    "video_keywords": _clean_list("video_keywords"),
                    "sfx_cues": _clean_list("sfx_cues"),
                    "raw_trends": raw_trends,
                }
    except Exception as exc:
        logger.warning(f"Failed to generate full viral video blueprint: {exc}")

    return {
        "success": False,
        "raw_trends": raw_trends,
        "error": "Could not generate viral blueprint from LLM.",
    }
