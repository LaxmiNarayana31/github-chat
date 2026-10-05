"""Centralized constants for GithubChat data scanning, file filtering, and indexing.

Provides categorized lists of directories, files, and file extensions that are
ignored or included during repository scanning and RAG preprocessing.
"""

from typing import List, Set

# ==============================================================================
# Directory Filtering
# ==============================================================================

# Directories that should never be scanned for code / docs
IGNORED_DIRS: Set[str] = {
    # Version Control
    ".git",
    ".github",
    ".gitlab",
    ".svn",
    ".hg",
    # Virtual Environments & Package Managers
    ".venv",
    "venv",
    "env",
    ".env",
    "virtualenv",
    ".conda",
    "node_modules",
    "bower_components",
    "jspm_packages",
    "vendor",
    # Caches & Runtimes
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "cache",
    ".cache",
    ".parcel-cache",
    ".turbo",
    ".next",
    ".nuxt",
    ".svelte-kit",
    # Build & Distribution Artifacts
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    "coverage",
    ".nyc_output",
    "htmlcov",
    # IDEs, Editors & OS Files
    ".idea",
    ".vscode",
    ".vs",
    ".eclipse",
    ".DS_Store",
    "Thumbs.db",
    ".local",
    "tmp",
    "temp",
    # Mega-Repository Noise & Non-Target Hardware Directories (Linux / Monorepos)
    "firmware",
    "staging",
    "samples",
    "translations",
    "csky",
    "m68k",
    "microblaze",
    "nios2",
    "parisc",
    "xtensa",
}

# ==============================================================================
# File Filtering
# ==============================================================================

# Large machine-generated metadata, lockfiles, minified files, or binaries that pollute context
IGNORED_FILES: Set[str] = {
    # Lockfiles (high-token, machine-generated noise)
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "uv.lock",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "mix.lock",
    # Secret / Environment configuration files
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    # OS artifacts
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
}

# File extensions for minified or bundled assets and firmware binaries to skip
IGNORED_FILE_SUFFIXES: tuple = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".bundle.css",
    ".map",
    # Binary blobs & compiled device trees common in C/C++/Kernel projects
    ".fw",
    ".bin",
    ".hex",
    ".dtb",
    ".dtbo",
    ".dts",
    ".o",
    ".a",
    ".so",
    ".dylib",
    ".dll",
)

# ==============================================================================
# Target File Extensions for Code and Documentation Indexing
# ==============================================================================

CODE_EXTENSIONS: List[str] = [
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".sql",
    ".sh",
    ".bash",
    ".lua",
    ".scala",
    ".dart",
]

DOC_EXTENSIONS: List[str] = [
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
]

ALL_INDEXABLE_EXTENSIONS: List[str] = CODE_EXTENSIONS + DOC_EXTENSIONS

# ==============================================================================
# Size and Length Thresholds
# ==============================================================================

# File size threshold capped at 250 KB to avoid context bloat and token overflows
MAX_FILE_SIZE_BYTES: int = 250 * 1024

# Minimum characters for a file to be indexed (avoids empty stubs or placeholder files)
MIN_FILE_CHAR_LENGTH: int = 10

