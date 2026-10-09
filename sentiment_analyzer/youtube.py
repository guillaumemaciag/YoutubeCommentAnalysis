"""Small, dependency-light client for the YouTube Data API v3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse
import re

import requests


API_ROOT = "https://www.googleapis.com/youtube/v3"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeAPIError(RuntimeError):
    """A readable error returned by the YouTube Data API."""


def extract_video_id(value: str) -> str:
    """Extract a YouTube video ID from a URL or accept a bare video ID."""
    value = value.strip()
    if VIDEO_ID_PATTERN.fullmatch(value):
        return value

    candidate_url = value if "://" in value else f"https://{value}"
    parsed = urlparse(candidate_url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]

    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
                video_id = parts[1]

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")
    return video_id


def _request_json(
    resource: str,
    params: dict[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    try:
        response = client.get(f"{API_ROOT}/{resource}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        message = "YouTube rejected the request. Check the API key and video permissions."
        try:
            payload = response.json()
            error = payload.get("error", {})
            message = error.get("message", message)
            details = error.get("errors", [])
            if details and details[0].get("reason") == "commentsDisabled":
                message = "Comments are disabled for this video."
        except (ValueError, AttributeError, TypeError):
            pass
        raise YouTubeAPIError(message) from exc
    except (requests.RequestException, ValueError) as exc:
        raise YouTubeAPIError(f"Could not reach YouTube: {exc}") from exc


def fetch_video_info(
    video_id: str,
    api_key: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch the display metadata and public statistics for a video."""
    payload = _request_json(
        "videos",
        {"part": "snippet,statistics", "id": video_id, "key": api_key},
        session=session,
    )
    if not payload.get("items"):
        raise YouTubeAPIError("Video not found, private, or unavailable to this API key.")

    item = payload["items"][0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = next(
        (thumbnails[name]["url"] for name in ("maxres", "standard", "high", "medium", "default") if name in thumbnails),
        "",
    )
    return {
        "video_id": video_id,
        "title": snippet.get("title", "Untitled video"),
        "channel": snippet.get("channelTitle", "Unknown channel"),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
        "thumbnail": thumbnail,
        "views": int(statistics.get("viewCount", 0)),
        "likes": int(statistics.get("likeCount", 0)),
        "comment_count": int(statistics.get("commentCount", 0)),
    }


def _parse_comment(item: dict[str, Any], *, is_reply: bool, parent_id: str = "") -> dict[str, Any]:
    snippet = item.get("snippet", {})
    return {
        "comment_id": item.get("id", ""),
        "parent_id": parent_id or snippet.get("parentId", ""),
        "is_reply": is_reply,
        "author": snippet.get("authorDisplayName", "Anonymous"),
        "text": snippet.get("textOriginal") or snippet.get("textDisplay", ""),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "likes": int(snippet.get("likeCount", 0)),
        "reply_count": 0,
    }


def _fetch_remaining_replies(
    parent_id: str,
    api_key: str,
    already_seen: set[str],
    limit: int,
    session: requests.Session,
) -> list[dict[str, Any]]:
    replies: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(replies) < limit:
        params: dict[str, Any] = {
            "part": "snippet",
            "parentId": parent_id,
            "maxResults": min(100, limit - len(replies)),
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        payload = _request_json("comments", params, session=session)
        for item in payload.get("items", []):
            if item.get("id") not in already_seen:
                replies.append(_parse_comment(item, is_reply=True, parent_id=parent_id))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return replies[:limit]


def fetch_comments(
    video_id: str,
    api_key: str,
    *,
    max_comments: int = 500,
    include_replies: bool = True,
    order: str = "relevance",
    session: requests.Session | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Fetch top-level comments and, optionally, replies up to ``max_comments``."""
    if max_comments < 1:
        return []
    if order not in {"relevance", "time"}:
        raise ValueError("order must be 'relevance' or 'time'")

    client = session or requests.Session()
    comments: list[dict[str, Any]] = []
    page_token: str | None = None

    while len(comments) < max_comments:
        params: dict[str, Any] = {
            "part": "snippet,replies" if include_replies else "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": order,
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        payload = _request_json("commentThreads", params, session=client)

        for thread in payload.get("items", []):
            if len(comments) >= max_comments:
                break
            thread_snippet = thread.get("snippet", {})
            top_level = thread_snippet.get("topLevelComment", {})
            parsed = _parse_comment(top_level, is_reply=False)
            parsed["reply_count"] = int(thread_snippet.get("totalReplyCount", 0))
            comments.append(parsed)

            if not include_replies or len(comments) >= max_comments:
                continue
            embedded = thread.get("replies", {}).get("comments", [])
            seen_ids: set[str] = set()
            for reply in embedded:
                if len(comments) >= max_comments:
                    break
                seen_ids.add(reply.get("id", ""))
                comments.append(_parse_comment(reply, is_reply=True, parent_id=parsed["comment_id"]))

            total_replies = parsed["reply_count"]
            if total_replies > len(embedded) and len(comments) < max_comments:
                remaining = min(total_replies - len(embedded), max_comments - len(comments))
                comments.extend(
                    _fetch_remaining_replies(
                        parsed["comment_id"], api_key, seen_ids, remaining, client
                    )
                )

        if progress:
            progress(min(len(comments), max_comments))
        page_token = payload.get("nextPageToken")
        if not page_token or not payload.get("items"):
            break

    return comments[:max_comments]
