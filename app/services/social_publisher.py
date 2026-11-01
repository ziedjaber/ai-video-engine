"""
Multi-Platform Auto-Publishing Service for MoneyPrinterTurbo.

Supports automatic and manual publishing of generated short-form videos to:
1. YouTube Shorts (Direct YouTube Data API v3 OAuth 2.0 OR Upload-Post)
2. Facebook Reels (Direct Meta Graph API /video_reels OR Upload-Post)
3. Instagram Reels (Direct Instagram Graph API resumable upload OR Upload-Post)
4. TikTok (Direct TikTok Content Posting API v2 OR Upload-Post)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services import llm, upload_post
from app.utils import utils

YOUTUBE_TOKEN_FILE = Path(utils.storage_dir()) / "youtube_oauth_token.json"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"

SUPPORTED_PUBLISH_PLATFORMS = (
    "youtube",
    "instagram",
    "facebook",
    "tiktok",
)

PLATFORM_DISPLAY_NAMES = {
    "youtube": "YouTube Shorts",
    "instagram": "Instagram Reels",
    "facebook": "Facebook Reels",
    "tiktok": "TikTok",
}


def get_youtube_token_path() -> Path:
    YOUTUBE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    return YOUTUBE_TOKEN_FILE


def has_youtube_oauth_token() -> bool:
    path = get_youtube_token_path()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("refresh_token") or data.get("access_token"))
    except Exception:
        return False


def save_youtube_oauth_token(token_data: dict[str, Any]) -> None:
    path = get_youtube_token_path()
    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    merged = {**existing, **token_data, "updated_at": int(time.time())}
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def authorize_youtube_local_server(port: int = 8000) -> tuple[bool, str]:
    """Run a local browser OAuth 2.0 flow using configured YouTube Client ID & Secret."""
    client_id = str(config.app.get("youtube_client_id", "") or "").strip()
    client_secret = str(config.app.get("youtube_client_secret", "") or "").strip()
    if not client_id or not client_secret:
        return False, "Missing YouTube Client ID or Client Secret in configuration."

    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.parse import parse_qs, urlencode, urlparse
        import webbrowser

        configured_redirect = str(config.app.get("youtube_redirect_uri", "") or "").strip()
        if configured_redirect.startswith("http://localhost") or configured_redirect.startswith("http://127.0.0.1"):
            parsed_redir = urlparse(configured_redirect)
            port = parsed_redir.port or port
            redirect_uri = configured_redirect
        else:
            redirect_uri = f"http://localhost:{port}/auth/youtube/callback"
        auth_params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": YOUTUBE_UPLOAD_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(auth_params)}"
        captured: dict[str, str] = {}

        class OAuthCallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                if "code" in qs:
                    captured["code"] = qs["code"][0]
                    body = b"<html><body style='font-family:sans-serif;text-align:center;padding:40px;'><h2>YouTube Connected Successfully!</h2><p>You can close this tab and return to MoneyPrinterTurbo.</p></body></html>"
                    self.send_response(200)
                else:
                    captured["error"] = qs.get("error", ["authorization_cancelled"])[0]
                    body = b"<html><body style='font-family:sans-serif;text-align:center;padding:40px;'><h2>YouTube Authorization Failed</h2></body></html>"
                    self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = HTTPServer(("localhost", port), OAuthCallbackHandler)
        server.timeout = 120
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        if "code" not in captured:
            return False, f"OAuth did not return an authorization code ({captured.get('error', 'timeout')})."

        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": captured["code"],
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
        save_youtube_oauth_token(token_data)
        return True, "YouTube account authorized and token saved!"
    except Exception as exc:
        logger.warning(f"YouTube local OAuth flow failed: {exc}")
        return False, str(exc)


def discover_meta_page_and_instagram(
    user_or_page_token: str,
    app_cfg: dict | None = None,
) -> dict[str, Any]:
    """
    Given a Meta User Access Token or Page Access Token, automatically discover:
    - Facebook Page ID, Page Name, and Page Access Token
    - Connected Instagram Business Account ID & username
    Also exchanges short-lived token for a 60-day long-lived token if App ID & Secret are set.
    """
    cfg = app_cfg or config.app
    token = (user_or_page_token or "").strip()
    if not token:
        return {"success": False, "error": "Please enter a Facebook User or Page Access Token first."}

    app_id = str(cfg.get("facebook_app_id", "") or "").strip()
    app_secret = str(cfg.get("facebook_app_secret", "") or "").strip()

    # Optional: try exchanging for a long-lived token if app credentials exist
    if app_id and app_secret:
        try:
            exchange_resp = requests.get(
                "https://graph.facebook.com/v21.0/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "fb_exchange_token": token,
                },
                timeout=15,
            )
            if exchange_resp.status_code == 200:
                exchanged = exchange_resp.json().get("access_token")
                if exchanged:
                    token = str(exchanged)
        except Exception as exc:
            logger.debug(f"Meta long-lived token exchange skipped: {exc}")

    try:
        # 1. Try /me/accounts (if token is a User Access Token)
        acct_resp = requests.get(
            "https://graph.facebook.com/v21.0/me/accounts",
            params={
                "fields": "id,name,access_token,instagram_business_account{id,username}",
                "access_token": token,
            },
            timeout=20,
        )
        if acct_resp.status_code == 200:
            pages = acct_resp.json().get("data") or []
            if pages:
                # Prefer a page that has an instagram_business_account attached
                chosen = next((p for p in pages if p.get("instagram_business_account")), pages[0])
                ig_obj = chosen.get("instagram_business_account") or {}
                return {
                    "success": True,
                    "page_id": str(chosen.get("id", "")),
                    "page_name": str(chosen.get("name", "")),
                    "page_access_token": str(chosen.get("access_token") or token),
                    "instagram_business_id": str(ig_obj.get("id", "")),
                    "instagram_username": str(ig_obj.get("username", "")),
                    "all_pages_count": len(pages),
                }

        # 2. Fallback: token might already be a Page Access Token -> query /me directly
        me_resp = requests.get(
            "https://graph.facebook.com/v21.0/me",
            params={
                "fields": "id,name,instagram_business_account{id,username}",
                "access_token": token,
            },
            timeout=20,
        )
        me_resp.raise_for_status()
        me_data = me_resp.json()
        ig_obj = me_data.get("instagram_business_account") or {}
        return {
            "success": True,
            "page_id": str(me_data.get("id", "")),
            "page_name": str(me_data.get("name", "")),
            "page_access_token": token,
            "instagram_business_id": str(ig_obj.get("id", "")),
            "instagram_username": str(ig_obj.get("username", "")),
            "all_pages_count": 1,
        }
    except Exception as exc:
        return {"success": False, "error": f"Meta Graph API discovery failed: {exc}"}


def _get_valid_youtube_access_token(app_cfg: dict | None = None) -> str:
    cfg = app_cfg or config.app
    path = get_youtube_token_path()
    if not path.is_file():
        raise ValueError("YouTube OAuth token not found. Click 'Authorize YouTube' in Auto-Publish settings first.")

    token_data = json.loads(path.read_text(encoding="utf-8"))
    refresh_token = str(token_data.get("refresh_token", "") or "").strip()
    client_id = str(cfg.get("youtube_client_id", "") or "").strip()
    client_secret = str(cfg.get("youtube_client_secret", "") or "").strip()

    if refresh_token and client_id and client_secret:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if resp.status_code == 200:
            refreshed = resp.json()
            save_youtube_oauth_token(refreshed)
            return str(refreshed["access_token"])

    access_token = str(token_data.get("access_token", "") or "").strip()
    if access_token:
        return access_token
    raise ValueError("Could not refresh YouTube OAuth access token.")


def publish_to_youtube_direct(
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    privacy_status: str = "public",
    made_for_kids: bool = False,
    app_cfg: dict | None = None,
) -> dict[str, Any]:
    """Upload a video directly to YouTube Shorts via YouTube Data API v3."""
    access_token = _get_valid_youtube_access_token(app_cfg)
    short_title = title.strip()[:90]
    if "#shorts" not in short_title.lower():
        short_title = f"{short_title} #Shorts"[:100]

    metadata = {
        "snippet": {
            "title": short_title,
            "description": description[:4900],
            "tags": [t.lstrip("#") for t in (tags or ["shorts", "3d", "science"])][:15],
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": privacy_status if privacy_status in ("public", "private", "unlisted") else "public",
            "selfDeclaredMadeForKids": bool(made_for_kids),
        },
    }

    file_size = os.path.getsize(video_path)
    init_resp = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": "video/mp4",
        },
        data=json.dumps(metadata),
        timeout=60,
    )
    init_resp.raise_for_status()
    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        raise RuntimeError("YouTube resumable upload did not return a Location header.")

    with open(video_path, "rb") as vf:
        upload_resp = requests.put(
            upload_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
            data=vf,
            timeout=600,
        )
    upload_resp.raise_for_status()
    data = upload_resp.json()
    video_id = data.get("id", "")
    return {
        "success": True,
        "platform": "youtube",
        "video_id": video_id,
        "url": f"https://youtube.com/shorts/{video_id}" if video_id else "",
    }


def publish_to_facebook_reels_direct(
    video_path: str,
    description: str,
    app_cfg: dict | None = None,
) -> dict[str, Any]:
    """Upload a video directly to Facebook Page Reels via Meta Graph API."""
    cfg = app_cfg or config.app
    page_id = str(cfg.get("facebook_page_id", "") or "").strip()
    page_token = str(cfg.get("facebook_page_access_token", "") or "").strip()
    if not page_id or not page_token:
        raise ValueError("Missing Facebook Page ID or Page Access Token.")

    # Step 1: Initialize upload session
    start_resp = requests.post(
        f"https://graph.facebook.com/v21.0/{page_id}/video_reels",
        data={"upload_phase": "start", "access_token": page_token},
        timeout=30,
    )
    start_resp.raise_for_status()
    start_data = start_resp.json()
    video_id = start_data.get("video_id")
    upload_url = start_data.get("upload_url") or f"https://rupload.facebook.com/video-upload/v21.0/{video_id}"

    # Step 2: Upload binary
    file_size = os.path.getsize(video_path)
    with open(video_path, "rb") as vf:
        up_resp = requests.post(
            upload_url,
            headers={
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "file_size": str(file_size),
            },
            data=vf,
            timeout=600,
        )
    up_resp.raise_for_status()

    # Step 3: Finish and publish Reel
    finish_resp = requests.post(
        f"https://graph.facebook.com/v21.0/{page_id}/video_reels",
        data={
            "access_token": page_token,
            "video_id": video_id,
            "upload_phase": "finish",
            "video_state": "PUBLISHED",
            "description": description[:2200],
        },
        timeout=60,
    )
    finish_resp.raise_for_status()
    return {
        "success": True,
        "platform": "facebook",
        "video_id": video_id,
        "url": f"https://www.facebook.com/reel/{video_id}" if video_id else "",
    }


def publish_to_instagram_reels_direct(
    video_path: str,
    caption: str,
    app_cfg: dict | None = None,
) -> dict[str, Any]:
    """Upload a local video directly to Instagram Business Reels via resumable Graph API."""
    cfg = app_cfg or config.app
    ig_id = str(cfg.get("instagram_business_id", "") or "").strip()
    access_token = str(cfg.get("facebook_page_access_token", "") or "").strip()
    if not ig_id or not access_token:
        raise ValueError("Missing Instagram Business ID or Facebook Page Access Token.")

    # Step 1: Create resumable Reels container
    init_resp = requests.post(
        f"https://graph.facebook.com/v21.0/{ig_id}/media",
        data={
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption[:2200],
            "access_token": access_token,
        },
        timeout=30,
    )
    init_resp.raise_for_status()
    init_data = init_resp.json()
    container_id = init_data.get("id")
    upload_uri = init_data.get("uri") or f"https://rupload.facebook.com/ig-api-upload/v21.0/{container_id}"

    # Step 2: Upload binary to rupload
    file_size = os.path.getsize(video_path)
    with open(video_path, "rb") as vf:
        up_resp = requests.post(
            upload_uri,
            headers={
                "Authorization": f"OAuth {access_token}",
                "offset": "0",
                "file_size": str(file_size),
            },
            data=vf,
            timeout=600,
        )
    up_resp.raise_for_status()

    # Step 3: Poll container status until FINISHED
    for _ in range(30):
        status_resp = requests.get(
            f"https://graph.facebook.com/v21.0/{container_id}",
            params={"fields": "status_code", "access_token": access_token},
            timeout=20,
        )
        if status_resp.status_code == 200:
            code = status_resp.json().get("status_code")
            if code == "FINISHED":
                break
            if code in ("ERROR", "EXPIRED"):
                raise RuntimeError(f"Instagram media container failed with status: {code}")
        time.sleep(4)

    # Step 4: Publish container
    pub_resp = requests.post(
        f"https://graph.facebook.com/v21.0/{ig_id}/media_publish",
        data={"creation_id": container_id, "access_token": access_token},
        timeout=60,
    )
    pub_resp.raise_for_status()
    media_id = pub_resp.json().get("id", "")
    return {
        "success": True,
        "platform": "instagram",
        "video_id": media_id,
    }


def get_platform_readiness(app_cfg: dict | None = None) -> dict[str, dict[str, Any]]:
    """Return connection readiness for each social platform."""
    cfg = app_cfg or config.app
    has_upload_post = bool(
        str(cfg.get("upload_post_api_key", "") or "").strip()
        and str(cfg.get("upload_post_username", "") or "").strip()
    )
    yt_oauth_ready = has_youtube_oauth_token() and bool(
        str(cfg.get("youtube_client_id", "") or "").strip()
    )
    fb_direct_ready = bool(
        str(cfg.get("facebook_page_id", "") or "").strip()
        and str(cfg.get("facebook_page_access_token", "") or "").strip()
    )
    ig_direct_ready = bool(
        str(cfg.get("instagram_business_id", "") or "").strip()
        and str(cfg.get("facebook_page_access_token", "") or "").strip()
    )
    tiktok_direct_ready = bool(str(cfg.get("tiktok_access_token", "") or "").strip())

    return {
        "youtube": {
            "ready": yt_oauth_ready or has_upload_post,
            "method": "YouTube OAuth 2.0" if yt_oauth_ready else ("Upload-Post" if has_upload_post else "Needs OAuth Connect"),
        },
        "facebook": {
            "ready": fb_direct_ready or has_upload_post,
            "method": "Meta Graph API" if fb_direct_ready else ("Upload-Post" if has_upload_post else "Needs Page Token & ID"),
        },
        "instagram": {
            "ready": ig_direct_ready or has_upload_post,
            "method": "Instagram Graph API" if ig_direct_ready else ("Upload-Post" if has_upload_post else "Needs IG Business ID & Token"),
        },
        "tiktok": {
            "ready": tiktok_direct_ready or has_upload_post,
            "method": "TikTok API" if tiktok_direct_ready else ("Upload-Post" if has_upload_post else "Needs Upload-Post / TikTok Key"),
        },
    }


def publish_video_multiplatform(
    video_path: str,
    video_subject: str,
    video_script: str,
    platforms: list[str] | None = None,
    language: str = "en-US",
    app_cfg: dict | None = None,
) -> dict[str, Any]:
    """
    Publish a generated video across all selected platforms using either direct APIs
    (YouTube OAuth, Facebook Graph, Instagram Graph) or Upload-Post fallback.
    """
    cfg = app_cfg or config.app
    selected_platforms = [
        p.strip().lower()
        for p in (platforms or cfg.get("upload_post_platforms") or ["youtube", "instagram", "facebook", "tiktok"])
        if str(p).strip()
    ]
    if not selected_platforms:
        return {"success": False, "error": "No target platforms selected", "results": {}}

    if not os.path.isfile(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}", "results": {}}

    # Generate AI social metadata (viral title, caption, hashtags)
    meta = llm.generate_social_metadata(
        video_subject=video_subject,
        video_script=video_script,
        language=language or "auto",
        platform="youtube_shorts" if "youtube" in selected_platforms else "tiktok",
    )
    title = meta.get("title") or video_subject or "3D Explainer #Shorts"
    caption = meta.get("caption") or video_script or title
    hashtags = meta.get("hashtags") or ["#shorts", "#3d", "#science", "#explainer"]
    hashtag_str = " ".join(hashtags)
    full_caption = f"{title}\n\n{caption}\n\n{hashtag_str}".strip()

    results: dict[str, Any] = {}
    upload_post_fallback_platforms: list[str] = []
    has_upload_post = bool(
        str(cfg.get("upload_post_api_key", "") or "").strip()
        and str(cfg.get("upload_post_username", "") or "").strip()
    )

    for platform in selected_platforms:
        try:
            if platform == "youtube" and has_youtube_oauth_token():
                results["youtube"] = publish_to_youtube_direct(
                    video_path=video_path,
                    title=title,
                    description=full_caption,
                    tags=hashtags,
                    privacy_status=str(cfg.get("upload_post_youtube_privacy_status", "public") or "public"),
                    made_for_kids=bool(cfg.get("upload_post_youtube_made_for_kids", False)),
                    app_cfg=cfg,
                )
            elif platform == "facebook" and str(cfg.get("facebook_page_id", "")).strip() and str(cfg.get("facebook_page_access_token", "")).strip():
                results["facebook"] = publish_to_facebook_reels_direct(
                    video_path=video_path,
                    description=full_caption,
                    app_cfg=cfg,
                )
            elif platform == "instagram" and str(cfg.get("instagram_business_id", "")).strip() and str(cfg.get("facebook_page_access_token", "")).strip():
                results["instagram"] = publish_to_instagram_reels_direct(
                    video_path=video_path,
                    caption=full_caption,
                    app_cfg=cfg,
                )
            elif has_upload_post:
                upload_post_fallback_platforms.append(platform)
            else:
                results[platform] = {
                    "success": False,
                    "error": f"{PLATFORM_DISPLAY_NAMES.get(platform, platform)} credentials not configured yet.",
                }
        except Exception as exc:
            logger.error(f"Direct publish to {platform} failed: {exc}")
            if has_upload_post:
                upload_post_fallback_platforms.append(platform)
            else:
                results[platform] = {"success": False, "error": str(exc)}

    if upload_post_fallback_platforms:
        up_res = upload_post.cross_post_video(
            video_path=video_path,
            title=full_caption[:2200],
            platforms=upload_post_fallback_platforms,
            youtube_extra={
                "youtube_title": title[:100],
                "youtube_description": full_caption[:4900],
                "tags": [t.lstrip("#") for t in hashtags],
                "privacyStatus": str(cfg.get("upload_post_youtube_privacy_status", "public") or "public"),
                "selfDeclaredMadeForKids": bool(cfg.get("upload_post_youtube_made_for_kids", False)),
            },
        )
        for p in upload_post_fallback_platforms:
            results[p] = {
                "success": bool(up_res.get("success")),
                "via": "upload-post",
                "details": up_res,
                "error": up_res.get("error", ""),
            }

    overall_success = bool(results) and all(r.get("success") for r in results.values())
    return {
        "success": overall_success,
        "metadata": {"title": title, "caption": full_caption, "hashtags": hashtags},
        "results": results,
    }
