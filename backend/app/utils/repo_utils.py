import hashlib
import ipaddress
import logging
import os
import re
import socket
import subprocess
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP string is a private, loopback, link-local, or cloud metadata address."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_safe_git_url(
    url_or_path: str, allow_local: bool | None = None
) -> tuple[bool, str]:
    """
    Enterprise SSRF and Command Injection validator for repository URLs and paths.
    Enforces:
    - Command option injection protection (disallows leading '-')
    - Protocol whitelisting (https://, http://, git@)
    - Rejection of file://, gopher://, and dangerous URI schemes
    - Rejection of cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    - DNS resolution verification rejecting private/loopback RFC 1918 IPs
    - Bounded length checks (max 512 chars)
    """
    if not url_or_path or not str(url_or_path).strip():
        return False, "Repository URL cannot be empty."

    cleaned = str(url_or_path).strip()

    # Guard 1: Command line option injection (e.g. --upload-pack=...)
    if cleaned.startswith("-"):
        return False, "Repository URL cannot start with a hyphen or CLI flag."

    # Guard 2: Maximum length check to prevent buffer/ReDoS issues
    if len(cleaned) > 512:
        return False, "Repository URL exceeds maximum allowed length of 512 characters."

    # Guard 3: Reject explicit forbidden schemes
    forbidden_schemes = ("file:", "gopher:", "ftp:", "dict:", "ldap:", "tftp:")
    if any(cleaned.lower().startswith(s) for s in forbidden_schemes):
        return False, "Unsupported or unsafe protocol scheme."

    # Guard 4: Local filesystem path handling
    if is_local_path(cleaned):
        permitted = (
            allow_local
            if allow_local is not None
            else os.getenv("ALLOW_LOCAL_REPOS", "true").lower() in ("true", "1", "yes")
        )
        if not permitted:
            return (
                False,
                "Local filesystem repository paths are disabled in this environment.",
            )
        return True, "Valid local repository path."

    # Guard 5: Remote URL validation and SSRF checking
    # Normalize for parsing
    test_url = cleaned if "://" in cleaned else f"https://{cleaned}"
    try:
        parsed = urlparse(test_url)
    except Exception as parse_err:
        return False, f"Malformed repository URL: {parse_err}"

    if parsed.scheme.lower() not in ("http", "https"):
        return (
            False,
            f"Invalid URL scheme '{parsed.scheme}'. Only HTTPS/HTTP are permitted.",
        )

    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        return False, "Repository URL missing valid hostname."

    # Explicit cloud metadata hostnames
    blocked_hosts = {
        "169.254.169.254",
        "169.254.170.2",
        "metadata.google.internal",
        "instance-data",
        "localhost",
        "127.0.0.1",
        "::1",
    }
    if (
        hostname in blocked_hosts
        or hostname.endswith(".internal")
        or hostname.endswith(".local")
    ):
        return False, f"Access to internal/metadata host '{hostname}' is blocked."

    # Direct IP check
    if is_private_ip(hostname):
        return False, f"Access to private IP range '{hostname}' is blocked."

    # Resolve hostname to verify no private IP DNS rebinding SSRF
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_candidate = str(sockaddr[0])
            if is_private_ip(ip_candidate):
                return (
                    False,
                    f"Hostname '{hostname}' resolves to blocked private address '{ip_candidate}'.",
                )
    except socket.gaierror:
        # If hostname does not resolve right now, allow valid standard domains like github.com
        if hostname not in ("github.com", "gitlab.com", "bitbucket.org"):
            log.debug(f"Could not resolve hostname '{hostname}' during SSRF check.")

    return True, "Valid remote repository URL."


def is_local_path(url_or_path: str) -> bool:
    """Check if the provided input is a local directory or file path."""
    if not url_or_path:
        return False
    # Check for drive letters (e.g. C:\, D:/), leading slashes (Linux/macOS), or existing filesystem paths
    if (
        re.match(r"^[a-zA-Z]:[/\\]", url_or_path)
        or url_or_path.startswith("/")
        or url_or_path.startswith("./")
        or url_or_path.startswith("../")
    ):
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
        cleaned = cleaned.removesuffix(".git")
        cleaned = cleaned.rstrip("/")

        return cleaned
    except Exception as e:
        log.warning(f"Error normalizing repo url '{url_or_path}': {e}")
        return str(url_or_path).strip()


def extract_repo_metadata(url_or_path: str) -> dict[str, Any]:
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


def get_authenticated_clone_url(repo_url: str, token: str | None = None) -> str:
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
        return normalized.replace(
            "https://github.com/", f"https://{auth_token}@github.com/"
        )

    return normalized


def get_git_commit_sha(repo_url_or_path: str, token: str | None = None) -> str | None:
    """Retrieve the latest commit SHA for a remote GitHub repository or local directory.

    For remote repos, executes 'git ls-remote <url> HEAD' in ~100-300ms without cloning.
    For local directories, executes 'git rev-parse HEAD'.
    """
    normalized = normalize_repo_url(repo_url_or_path)
    if not normalized:
        return None

    if is_local_path(normalized):
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=normalized,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            return None

    is_safe, reason = validate_safe_git_url(normalized)
    if not is_safe:
        log.warning(
            f"get_git_commit_sha: Blocked unsafe repository URL '{normalized}': {reason}"
        )
        return None

    try:
        clone_url = get_authenticated_clone_url(normalized, token=token)
        res = subprocess.run(
            ["git", "ls-remote", "--", clone_url, "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0]
            sha = first_line.split()[0].strip()
            if len(sha) >= 7:
                return sha
    except Exception as e:
        log.debug(f"git ls-remote failed for {normalized}: {e}")

    return None
