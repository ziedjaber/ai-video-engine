from typing import Optional
from fastapi import Depends, Request
from pydantic import BaseModel

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import llm, social_publisher, trend_discovery
from app.utils import utils

# LLM 接口与视频接口共用同一鉴权规则，避免新增端点时遗漏保护。
# api_key 为空时 verify_token 直接放行，不改变默认本地使用体验。
router = new_router(dependencies=[Depends(base.verify_token)])


class TrendDiscoveryRequest(BaseModel):
    sources: Optional[list[str]] = None
    count: int = 5
    language: str = "en-US"
    custom_query_prefix: str = ""


class MultiPlatformPublishRequest(BaseModel):
    video_path: str
    video_subject: str = ""
    video_script: str = ""
    platforms: Optional[list[str]] = None
    language: str = "en-US"


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Create a script for the video",
)
def generate_video_script(request: Request, body: VideoScriptRequest):
    video_script = llm.generate_script(
        video_subject=body.video_subject,
        language=body.video_language,
        paragraph_number=body.paragraph_number,
        video_script_prompt=body.video_script_prompt,
        custom_system_prompt=body.custom_system_prompt,
    )
    if not video_script or video_script.startswith("Error: "):
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="script generation is unavailable",
        )
    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generate video terms based on the video script",
)
def generate_video_terms(request: Request, body: VideoTermsRequest):
    video_terms = llm.generate_terms(
        video_subject=body.video_subject,
        video_script=body.video_script,
        amount=body.amount,
        match_script_order=body.match_materials_to_script,
    )
    if not video_terms:
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="search term generation is unavailable",
        )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generate social publishing metadata",
)
def generate_video_social_metadata(
    request: Request, body: VideoSocialMetadataRequest
):
    metadata = llm.generate_social_metadata(
        video_subject=body.video_subject,
        video_script=body.video_script,
        language=body.language,
        platform=body.platform,
    )
    return utils.get_response(200, metadata)


@router.post(
    "/trends/discover",
    summary="Fetch live trends (Google Autocomplete, Reddit, Hacker News) and curate Halal-friendly viral topics",
)
def discover_viral_trends(request: Request, body: TrendDiscoveryRequest):
    data = trend_discovery.generate_viral_topics_from_trends(
        sources=body.sources,
        count=body.count,
        language=body.language,
        custom_query_prefix=body.custom_query_prefix,
    )
    return utils.get_response(200, data)


@router.post(
    "/trends/blueprint",
    summary="Fetch live trends, inject {{TRENDING_TOPICS_ARRAY}}, and generate a complete viral video blueprint (topic, script, 3D scenes, SFX)",
)
def generate_viral_blueprint_endpoint(request: Request, body: TrendDiscoveryRequest):
    data = trend_discovery.generate_viral_video_blueprint(
        sources=body.sources,
        language=body.language,
        custom_query_prefix=body.custom_query_prefix,
    )
    return utils.get_response(200, data)


@router.post(
    "/social/publish",
    summary="Auto-publish a generated video to YouTube Shorts, Instagram Reels, Facebook Reels, and TikTok",
)
def publish_video_social_endpoint(request: Request, body: MultiPlatformPublishRequest):
    outcome = social_publisher.publish_video_multiplatform(
        video_path=body.video_path,
        video_subject=body.video_subject,
        video_script=body.video_script,
        platforms=body.platforms,
        language=body.language,
    )
    return utils.get_response(200, outcome)
