import os
import pytest

from backend.app.utils.repo_utils import (
    extract_repo_metadata,
    get_authenticated_clone_url,
    get_repo_slug,
    is_local_path,
    normalize_repo_url,
    sanitize_collection_slug,
)


def test_normalize_repo_url_variations():
    """Verify various GitHub URL formats normalize to standard canonical URLs."""
    assert normalize_repo_url("https://github.com/facebook/react") == "https://github.com/facebook/react"
    assert normalize_repo_url("https://github.com/facebook/react.git") == "https://github.com/facebook/react"
    assert normalize_repo_url("https://github.com/facebook/react/") == "https://github.com/facebook/react"
    assert normalize_repo_url("github.com/facebook/react") == "https://github.com/facebook/react"
    assert normalize_repo_url("https://github.com/facebook/react/tree/main") == "https://github.com/facebook/react"
    assert normalize_repo_url("https://github.com/facebook/react/blob/main/README.md") == "https://github.com/facebook/react"
    assert normalize_repo_url("git@github.com:facebook/react.git") == "https://github.com/facebook/react"


def test_collision_free_slugs():
    """Verify different repositories with the same repo name have completely distinct slugs."""
    slug_fb = get_repo_slug("https://github.com/facebook/react")
    slug_myorg = get_repo_slug("https://github.com/myorg/react")

    assert slug_fb != slug_myorg
    assert "facebook_react" in slug_fb
    assert "myorg_react" in slug_myorg


def test_deterministic_slug():
    """Verify that calling get_repo_slug on the same repo always produces the identical slug."""
    slug1 = get_repo_slug("https://github.com/astral-sh/uv")
    slug2 = get_repo_slug("github.com/astral-sh/uv.git")
    assert slug1 == slug2


def test_sanitize_collection_slug():
    """Verify Qdrant collection naming constraints (<= 64 chars, prefix, valid chars)."""
    slug = get_repo_slug("https://github.com/very-long-organization-name-12345/very-long-repository-name-67890")
    col_name = sanitize_collection_slug(slug)
    assert len(col_name) <= 64
    assert col_name.startswith("github_chat_")
    assert all(c.isalnum() or c in "_-" for c in col_name)


def test_authenticated_clone_url():
    """Verify GitHub token is properly injected into clone URL."""
    url = "https://github.com/private-org/secret-repo"
    auth_url = get_authenticated_clone_url(url, token="ghp_testToken12345")
    assert "ghp_testToken12345@github.com" in auth_url

    # When no token provided, return standard URL
    no_token_url = get_authenticated_clone_url(url, token=None)
    assert "@" not in no_token_url


def test_local_path_handling():
    """Verify local folder paths are recognized and handled properly."""
    local_dir = os.path.dirname(os.path.abspath(__file__))
    assert is_local_path(local_dir) is True
    meta = extract_repo_metadata(local_dir)
    assert meta["is_local"] is True
    slug = get_repo_slug(local_dir)
    assert "local" in slug

