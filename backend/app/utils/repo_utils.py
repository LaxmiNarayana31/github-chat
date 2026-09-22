import hashlib
import logging
import os
import re
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def is_local_path(url_or_path: str) -> bool:
    """Check if the provided input is a local directory or file path."""
    if not url_or_path:
        return False
    # Check for drive letters (e.g. C:\, D:/), leading slashes (Linux/macOS), or existing filesystem paths
    if re.match(r"^[a-zA-Z]:[/\\]", url_or_path) or url_or_path.startswith("/") or url_or_path.startswith("./") or url_or_path.startswith("../"):
        return True
    return os.path.exists(url_or_path)


def normalize_repo_url(url_or_path: str) -> str:
    """
    Sanitize and normalize a GitHub repository URL or local directory path.
    Handles:
    - Missing protocol: 'github.com/owner/repo' -> 'https://github.com/owner/repo'
    - Tree/Blob paths: 'https://github.com/owner/repo/tree/main' -> 'https://github.com/owner/repo'
    - SSH URLs: 'git@github.com:owner/repo.git' -> 'https://github.com/owner/repo'
    - Local filesystem paths: 'D:\\path\\to\\repo' -> 'D:/path/to/repo'
    """
    try:
        if not url_or_path:
            return ""
        cleaned = str(url_or_path).strip()

        # Handle local filesystem path
        if is_local_path(cleaned):
            return os.path.abspath(cleaned).replace("\\", "/")

        # Handle SSH URLs (git@github.com:owner/repo.git)
        ssh_match = re.match(r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", cleaned)
        if ssh_match:
            owner, repo = ssh_match.group(1), ssh_match.group(2)
            return f"https://github.com/{owner}/{repo}"

        # If missing protocol but is github.com, prepend https://
        if cleaned.startswith("github.com/"):
            cleaned = f"https://{cleaned}"

        # Clean git branches or subpaths like /tree/main, /blob/master, /commit/...
        cleaned = re.sub(r"/tree/[^/]+.*$", "", cleaned)
        cleaned = re.sub(r"/blob/[^/]+.*$", "", cleaned)

        # Remove trailing .git and trailing slashes
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]
        cleaned = cleaned.rstrip("/")

        return cleaned
    except Exception as e:
        log.warning(f"Error normalizing repo url '{url_or_path}': {e}")
        return str(url_or_path).strip()


def extract_repo_metadata(url_or_path: str) -> Dict[str, Any]:
    """Extract owner, repository name, and canonical attributes."""
    normalized = normalize_repo_url(url_or_path)
    if is_local_path(normalized):
        folder_name = os.path.basename(os.path.normpath(normalized)) or "local_repo"
        return {
            "owner": "local",
            "repo": folder_name,
            "is_local": True,
            "canonical_url": normalized,
        }

    # Match standard GitHub URL: https://github.com/owner/repo
    gh_match = re.search(r"github\.com/([^/]+)/([^/]+)", normalized)
    if gh_match:
        owner = gh_match.group(1).strip()
        repo = gh_match.group(2).strip()
        return {
            "owner": owner,
            "repo": repo,
            "is_local": False,
            "canonical_url": f"https://github.com/{owner}/{repo}",
        }

    # Fallback for other git remotes
    parts = [p for p in normalized.split("/") if p]
    repo_name = parts[-1] if parts else "unknown_repo"
    owner_name = parts[-2] if len(parts) > 1 else "external"
    return {
        "owner": owner_name,
        "repo": repo_name,
        "is_local": False,
        "canonical_url": normalized,
    }


def get_repo_slug(url_or_path: str) -> str:
    """
    Generate a deterministic, collision-free slug for a repository.
    Format: '{owner}_{repo}_{sha256[:8]}'
    Prevents collision between e.g. facebook/react and myorg/react.
    """
    meta = extract_repo_metadata(url_or_path)
    canonical = meta["canonical_url"].lower().strip()
    url_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    raw_owner = re.sub(r"[^a-zA-Z0-9_-]", "_", meta["owner"]).strip("_").lower()
    raw_repo = re.sub(r"[^a-zA-Z0-9_-]", "_", meta["repo"]).strip("_").lower()

    # Limit lengths to ensure entire slug fits within bounds
    owner_part = raw_owner[:20]
    repo_part = raw_repo[:24]

    return f"{owner_part}_{repo_part}_{url_hash}"


def sanitize_collection_slug(slug_or_name: str, prefix: str = "github_chat_") -> str:
    """
    Produce a valid Qdrant collection name adhering strictly to:
    - Maximum 64 characters
    - Only [a-z0-9_-] characters
    """
    # If already starts with prefix, don't duplicate
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", slug_or_name).strip("_").lower()
    if clean.startswith(prefix):
        full = clean
    else:
        full = f"{prefix}{clean}"
    return full[:64]


def get_authenticated_clone_url(repo_url: str, token: Optional[str] = None) -> str:
    """
    Inject GitHub Personal Access Token (PAT) into clone URL for private repositories.
    Never logs the token.
    """
    normalized = normalize_repo_url(repo_url)
    if is_local_path(normalized):
        return normalized

    auth_token = token or os.getenv("GITHUB_TOKEN", "").strip()
    if not auth_token:
        return normalized

    # Inject token into https://github.com/owner/repo
    if "github.com/" in normalized and not ("@" in normalized.split("github.com")[0]):
        return normalized.replace("https://github.com/", f"https://{auth_token}@github.com/")

    return normalized
