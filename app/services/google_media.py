"""
Google Media Engine (Nano Banana 2.1 / Pro & Veo 3.1) for MoneyPrinterTurbo.

Uses the official Google GenAI SDK (`google-genai`) and your Google AI Studio API key
to generate ultra-clean, watermark-free 3D educational cutaways and motion videos
in the style of Zack D. Films.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
from PIL import Image

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def get_google_api_key() -> str:
    """Retrieve the Gemini / Google AI Studio API key from env or configuration."""
    key = (
        str(os.getenv("GEMINI_API_KEY", "") or "").strip()
        or str(config.app.get("gemini_api_key", "") or "").strip()
        or str(os.getenv("GOOGLE_API_KEY", "") or "").strip()
    )
    return key


def is_google_media_enabled() -> bool:
    """Check if the official Google GenAI SDK is available and an API key is configured."""
    return _GENAI_AVAILABLE and bool(get_google_api_key())


class GoogleMediaEngine:
    """
    Direct interface to Google AI Studio's image and video generation models.
    Supports:
    - Nano Banana 2.1 (gemini-nano-banana-2.1 / gemini-3.1-flash-image)
    - Nano Banana Pro (gemini-3-pro-image / gemini-3-pro-image-preview)
    - Flash Image (gemini-2.5-flash-image)
    - Veo 3.1 (veo-3.1-fast-generate-preview / veo-3.1-generate-preview / veo-2.0-generate-001)
    """

    NANO_BANANA_MODELS = (
        "gemini-nano-banana-2.1",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    )

    VEO_MODELS = (
        "veo-3.1-fast-generate-preview",
        "veo-3.1-generate-preview",
        "veo-2.0-generate-001",
    )

    def __init__(self, api_key: str | None = None):
        if not _GENAI_AVAILABLE:
            raise RuntimeError("The google-genai package is not installed. Run: pip install -U google-genai")
        self.api_key = (api_key or get_google_api_key()).strip()
        if not self.api_key:
            raise ValueError(
                "Missing Google AI Studio API key. Set GEMINI_API_KEY in your .env or config.toml"
            )
        self.client = genai.Client(api_key=self.api_key)

    def generate_nano_banana_image(
        self,
        prompt: str,
        output_path: str,
        model: str = "gemini-nano-banana-2.1",
        aspect_ratio: str = "9:16",
    ) -> str:
        """
        Generates a 9:16 3D cutaway image using Google Nano Banana (zero watermarks, ultra-crisp).
        Automatically tries fallback models if the requested variant is not available.
        """
        # Ensure Zack D. style direction is present
        enhanced_prompt = prompt.strip()
        if "zack d" not in enhanced_prompt.lower():
            enhanced_prompt = (
                f"{enhanced_prompt}. Bright 3D educational cutaway render in the style of Zack D Films, "
                "clean light-slate studio background, bright medical-grade studio lighting, "
                "sharp macro focus, Cinema4D Octane render, vertical 9:16 aspect ratio, no watermark, no text."
            )

        models_to_try = [model] + [m for m in self.NANO_BANANA_MODELS if m != model]
        last_error = None

        for model_candidate in models_to_try:
            try:
                logger.info(f"generating image with Google model '{model_candidate}' (9:16)...")
                response = self.client.models.generate_content(
                    model=model_candidate,
                    contents=[enhanced_prompt],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                        ),
                    ),
                )
                if response and hasattr(response, "parts") and response.parts:
                    for part in response.parts:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
                            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                            img = Image.open(BytesIO(inline_data.data))
                            img.save(output_path, format="PNG")
                            logger.success(f"Google Nano Banana image saved: {output_path} ({img.size})")
                            return output_path
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    logger.warning(f"Google {model_candidate} quota limit (Free Tier needs billing link in AI Studio): {exc}")
                    raise exc
                logger.debug(f"Google model '{model_candidate}' skipped: {exc}")

        raise RuntimeError(f"Google Nano Banana image generation failed: {last_error}")

    def generate_veo3_video(
        self,
        prompt: str,
        output_mp4_path: str,
        starting_image_path: str | None = None,
        model: str = "veo-3.1-fast-generate-preview",
        duration_seconds: int = 6,
    ) -> str:
        """
        Generates a real 9:16 3D animated video clip using Google Veo 3.1.
        Can animate a Nano Banana 3D image directly if starting_image_path is provided!
        """
        image_input = None
        if starting_image_path and os.path.exists(starting_image_path):
            image_input = Image.open(starting_image_path)

        veo_prompt = (
            f"{prompt}, smooth 3D camera zoom into the cutaway mechanism, "
            "Zack D Films style, clean 3D animation, mechanical movement"
        )

        logger.info(f"submitting Veo 3.1 video task to Google (model={model})...")
        operation = self.client.models.generate_videos(
            model=model,
            prompt=veo_prompt,
            image=image_input,
            config=types.GenerateVideosConfig(
                aspect_ratio="9:16",
                resolution="1080p",
                duration_seconds=duration_seconds,
                person_generation="dont_allow",  # Enforces 100% Halal faceless/object focus
            ),
        )

        # Poll until Veo 3.1 finishes rendering the 3D clip
        logger.info("waiting for Veo 3.1 video rendering to complete...")
        deadline = time.monotonic() + 600
        while not operation.done:
            if time.monotonic() > deadline:
                raise TimeoutError("Veo 3.1 video generation timed out after 10 minutes.")
            time.sleep(8)
            operation = self.client.operations.get(operation)

        if not hasattr(operation, "response") or not getattr(operation.response, "generated_videos", None):
            raise RuntimeError(f"Veo 3.1 completed without video output: {operation}")

        generated_video = operation.response.generated_videos[0]
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
        self.client.files.download(file=generated_video.video)
        generated_video.video.save(output_mp4_path)
        logger.success(f"Veo 3.1 video saved: {output_mp4_path}")
        return output_mp4_path


# Module-level singleton
_engine: Optional[GoogleMediaEngine] = None


def get_google_media_engine() -> GoogleMediaEngine:
    global _engine
    if _engine is None:
        _engine = GoogleMediaEngine()
    return _engine


def generate_images_google(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Generate an ultra-crisp 3D cutaway image using Google Nano Banana (9:16 vertical).
    Matches the MoneyPrinterTurbo material provider interface.
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    clip_duration = max(int(minimum_duration), 1)
    file_id = utils.md5(f"google-{search_term}-{time.time()}")
    output_path = os.path.join(save_dir, f"google-nano-{file_id}.png")

    aspect_str = "9:16" if video_aspect in (VideoAspect.portrait, "9:16", "portrait") else "16:9"
    engine = get_google_media_engine()
    saved_path = engine.generate_nano_banana_image(
        prompt=search_term,
        output_path=output_path,
        aspect_ratio=aspect_str,
    )

    with Image.open(saved_path) as img:
        width, height = img.size

    item = MaterialInfo()
    item.provider = "google_nano_banana"
    item.url = saved_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "google_nano_banana",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]
