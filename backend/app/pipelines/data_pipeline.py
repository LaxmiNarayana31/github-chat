import gc
import glob
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import adalflow as adal
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from adalflow.core.db import LocalDB
from adalflow.core.types import Document
from adalflow.utils import get_adalflow_default_root_path, printc
import httpx

from backend.app.config.config import config
from backend.app.config.constants import (
    ALL_INDEXABLE_EXTENSIONS,
    CODE_EXTENSIONS,
    IGNORED_DIRS,
    IGNORED_FILES,
    IGNORED_FILE_SUFFIXES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_CHAR_LENGTH,
)
from backend.app.pipelines.ast_chunker import ASTChunker
from backend.app.pipelines.blob_deduplicator import BlobDeduplicator
from backend.app.rag.repo_map import RepoMapGenerator
from backend.app.services.redis_manager import RedisCacheManager
from backend.app.utils.repo_utils import (
    extract_repo_metadata,
    get_authenticated_clone_url,
    get_repo_slug,
    sanitize_collection_slug,
)

log = logging.getLogger(__name__)


# Clone github repo to local path
def download_github_repo(repo_url: str, local_path: str, token: Optional[str] = None):
    """Clone a remote GitHub repository to a local directory with error handling and optional PAT token."""
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.makedirs(local_path, exist_ok=True)
        # If directory already has files, remove or re-clone
        if os.path.exists(local_path) and os.listdir(local_path):
            printc(f"Directory {local_path} already exists, clearing for fresh clone...", color="yellow")
            try:
                shutil.rmtree(local_path)
            except Exception as rm_err:
                printc(f"Could not clear {local_path}: {rm_err}", color="yellow")

        clone_url = get_authenticated_clone_url(repo_url, token=token)
        # Attempt blobless shallow clone (--filter=blob:none) to save gigabytes of bandwidth and disk on mega-repos
        cmd_blobless = ["git", "clone", "--depth", "1", "--single-branch", "--no-tags", "--filter=blob:none", clone_url, local_path]
        try:
            result = subprocess.run(
                cmd_blobless,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return result.stdout.decode("utf-8")
        except subprocess.CalledProcessError:
            # Fallback to standard shallow clone if git server does not support blobless filters
            cmd_standard = ["git", "clone", "--depth", "1", "--single-branch", "--no-tags", clone_url, local_path]
            result = subprocess.run(
                cmd_standard,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace")
        if token:
            err_msg = err_msg.replace(token, "******")
        printc(f"Git clone error: {err_msg}", color="red")
        return f"Error during cloning: {err_msg}"
    except Exception as e:
        err_str = str(e)
        if token:
            err_str = err_str.replace(token, "******")
        printc(f"Unexpected error during repo download: {err_str}", color="red")
        return f"Unexpected error: {err_str}"


def stream_github_tree_documents(
    repo_url: str,
    token: Optional[str] = None,
    branch: str = "HEAD",
    max_files: int = 1000,
) -> List[Document]:
    """Stream and parse indexable repository files directly via GitHub Git Trees API without local disk cloning."""
    meta = extract_repo_metadata(repo_url)
    if meta.get("is_local"):
        log.info(f"Local repo path detected, using local document reader: {repo_url}")
        return read_all_documents(repo_url)

    owner = meta.get("owner")
    repo = meta.get("repo")
    if not owner or not repo:
        log.warning(f"Could not extract owner/repo from URL: {repo_url}")
        return []

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubChat-Enterprise-Ingest/1.0",
    }
    auth_token = token or os.getenv("GITHUB_TOKEN") or config.get("github_token")
    if auth_token:
        headers["Authorization"] = f"token {auth_token}"

    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    documents: List[Document] = []

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(tree_url, headers=headers)
            if resp.status_code != 200:
                log.warning(f"GitHub Trees API error ({resp.status_code}): {resp.text[:200]}")
                return []

            tree_data = resp.json()
            items = tree_data.get("tree", [])

            indexable_items = []
            for item in items:
                if item.get("type") != "blob":
                    continue
                file_path = item.get("path", "")
                parts = set(os.path.normpath(file_path).split(os.sep))
                if parts & IGNORED_DIRS:
                    continue
                filename = os.path.basename(file_path).lower()
                if filename in IGNORED_FILES:
                    continue
                if any(filename.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES):
                    continue

                _, ext = os.path.splitext(filename)
                if ext.lower() not in ALL_INDEXABLE_EXTENSIONS:
                    continue

                size = item.get("size", 0)
                if size == 0 or size > MAX_FILE_SIZE_BYTES:
                    continue

                indexable_items.append(item)
                if len(indexable_items) >= max_files:
                    break

            log.info(f"GitHub Trees API: Discovered {len(indexable_items)} indexable files in {owner}/{repo}")

            for item in indexable_items:
                file_path = item.get("path", "")
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
                content_resp = client.get(raw_url, headers=headers)
                if content_resp.status_code != 200:
                    continue

                content = content_resp.text
                if len(content.strip()) < MIN_FILE_CHAR_LENGTH:
                    continue

                _, ext = os.path.splitext(file_path)
                ext_clean = ext.lower()
                is_code = ext_clean in CODE_EXTENSIONS

                documents.append(
                    Document(
                        text=content,
                        meta_data={
                            "file_path": file_path,
                            "type": ext_clean.lstrip("."),
                            "is_code": is_code,
                            "is_implementation": is_code,
                            "title": file_path,
                            "repo_url": repo_url,
                        },
                    )
                )

        return documents
    except Exception as e:
        log.warning(f"Error streaming GitHub tree for {repo_url}: {e}")
        return []


# Read all documents from local path with production-grade filtering
def read_all_documents(path: str) -> List[Document]:
    """Scan and read all valid code and documentation files, skipping binaries and lockfiles."""
    documents = []

    for ext in ALL_INDEXABLE_EXTENSIONS:
        for file_path in glob.glob(f"{path}/**/*{ext}", recursive=True):
            # Normalize path and check against ignored directories
            parts = set(os.path.normpath(file_path).split(os.sep))
            if parts & IGNORED_DIRS:
                continue

            filename = os.path.basename(file_path).lower()
            if filename in IGNORED_FILES:
                continue

            # Skip minified bundle files or ignored suffixes
            if any(filename.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES):
                continue

            # Check file size before reading
            try:
                size = os.path.getsize(file_path)
                if size == 0 or size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Ignore empty or near-empty files (e.g. empty __init__.py)
                stripped = content.strip()
                if len(stripped) < MIN_FILE_CHAR_LENGTH:
                    continue

                rel = os.path.relpath(file_path, path)
                is_code = ext in CODE_EXTENSIONS
                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": rel,
                        "type": ext[1:],
                        "is_code": is_code,
                        "is_implementation": is_code,
                        "title": rel,
                    },
                )
                documents.append(doc)
            except Exception as e:
                log.warning(f"Error reading {file_path}: {e}")

    printc(f"DataPipeline: Read {len(documents)} clean source files from {path}", color="green")
    return documents


def stream_documents(path: str):
    """Stream valid code and documentation files using generator traversal for constant memory usage."""
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for filename in files:
            file_lower = filename.lower()
            ext = os.path.splitext(file_lower)[1]
            if ext not in ALL_INDEXABLE_EXTENSIONS or file_lower in IGNORED_FILES:
                continue
            if any(file_lower.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES):
                continue

            file_path = os.path.join(root, filename)
            try:
                size = os.path.getsize(file_path)
                if size == 0 or size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                stripped = content.strip()
                if len(stripped) < MIN_FILE_CHAR_LENGTH:
                    continue
                rel = os.path.relpath(file_path, path)
                is_code = ext in CODE_EXTENSIONS
                yield Document(
                    text=content,
                    meta_data={
                        "file_path": rel,
                        "type": ext.lstrip("."),
                        "is_code": is_code,
                        "is_implementation": is_code,
                        "title": rel,
                    },
                )
            except Exception as e:
                log.warning(f"Error streaming {file_path}: {e}")


# Prepare data pipeline for embedding
def prepare_data_pipeline():
    """Standard sequential splitter + embedder pipeline."""
    splitter = TextSplitter(**config["text_splitter"])
    embedder = adal.Embedder(
        model_client=config["embedder"]["model_client"](),
        model_kwargs=config["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=config["embedder"]["batch_size"]
    )
    return adal.Sequential(splitter, embedder_transformer)


def prepare_embedding_pipeline():
    """Direct embedding transformer without TextSplitter for pre-chunked AST documents."""
    embedder = adal.Embedder(
        model_client=config["embedder"]["model_client"](),
        model_kwargs=config["embedder"]["model_kwargs"],
    )
    return ToEmbeddings(
        embedder=embedder, batch_size=config["embedder"]["batch_size"]
    )


# Transform documents, validate embedding dimensions, and save to db
def transform_documents_and_save_to_db(documents: List[Document], db_path: str) -> LocalDB:
    expected_dim = config["embedder"].get("dimensions", 768)
    ast_chunker = ASTChunker()
    dedup = BlobDeduplicator.get_instance()
    redis_mgr = RedisCacheManager.get_instance()

    # Step 1: Structural AST and syntax chunking with Git-blob deduplication
    docs_to_embed: List[Document] = []
    already_embedded_docs: List[Document] = []
    all_chunks_for_repo_map: List[Document] = []

    splitter_cfg = config.get("text_splitter", {})
    text_splitter = TextSplitter(**splitter_cfg) if splitter_cfg else None

    for doc in documents:
        meta = doc.meta_data or {}
        file_path = meta.get("file_path", "")
        file_type = meta.get("type", "txt")
        is_code = meta.get("is_code", False)

        # Compute Git blob SHA-1 hash for content-addressable caching
        blob_hash = dedup.compute_blob_hash(doc.text)
        cached_chunks = dedup.get_cached_chunks(blob_hash)
        if cached_chunks:
            for item in cached_chunks:
                c_doc = Document(
                    text=item["text"],
                    meta_data=item.get("meta_data", {}),
                )
                if "vector" in item and item["vector"] and len(item["vector"]) == expected_dim:
                    c_doc.vector = item["vector"]
                    already_embedded_docs.append(c_doc)
                else:
                    docs_to_embed.append(c_doc)
                all_chunks_for_repo_map.append(c_doc)
            continue

        if is_code:
            ast_chunks = ast_chunker.chunk_file(doc.text, file_path=file_path, file_type=file_type)
            if ast_chunks:
                for chunk in ast_chunks:
                    chunk_meta = chunk.meta_data or {}
                    chunk_meta["blob_hash"] = blob_hash
                    c = Document(text=chunk.text, meta_data=chunk_meta)
                    docs_to_embed.append(c)
                    all_chunks_for_repo_map.append(c)
                continue

        # Fallback for non-code files (Markdown, text docs)
        if doc.meta_data is None:
            doc.meta_data = {}
        doc.meta_data["blob_hash"] = blob_hash

        # For non-code docs exceeding chunk size, split cleanly without AST
        chunk_size = splitter_cfg.get("chunk_size", 400) if splitter_cfg else 400
        if len(doc.text) > chunk_size and text_splitter:
            split_chunks = text_splitter.call([doc])
            for sc in split_chunks:
                sc_meta = getattr(sc, "meta_data", {}) or {}
                sc_meta["blob_hash"] = blob_hash
                c = Document(text=sc.text, meta_data=sc_meta)
                docs_to_embed.append(c)
                all_chunks_for_repo_map.append(c)
        else:
            docs_to_embed.append(doc)
            all_chunks_for_repo_map.append(doc)

    # Step 2: Build and cache PageRank Repository Map
    try:
        repo_map_gen = RepoMapGenerator().build_from_chunks(all_chunks_for_repo_map)
        rendered_map = repo_map_gen.generate_repo_map()
        slug = os.path.basename(db_path).replace(".pkl", "")
        if redis_mgr and redis_mgr.is_connected and rendered_map:
            redis_mgr.setex(f"repomap:{slug}", 86400 * 7, rendered_map)
            col_name = sanitize_collection_slug(slug)
            redis_mgr.setex(f"repomap:{col_name}", 86400 * 7, rendered_map)
            log.info(f"DataPipeline: Cached PageRank Repo Map for '{slug}' & '{col_name}' ({len(rendered_map)} chars).")
    except Exception as map_err:
        log.warning(f"DataPipeline: Error building RepoMap: {map_err}")

    # Step 3: Embed documents directly via ToEmbeddings (avoiding character-based re-splitting)
    newly_embedded_docs: List[Document] = []
    db = LocalDB()
    if docs_to_embed:
        embedder_transformer = prepare_embedding_pipeline()
        db.register_transformer(transformer=embedder_transformer, key="split_and_embed")
        db.load(docs_to_embed)
        db.transform(key="split_and_embed")
        newly_embedded_docs = db.get_transformed_data(key="split_and_embed") or []

    transformed_docs = list(already_embedded_docs) + list(newly_embedded_docs)
    if not transformed_docs:
        printc("No transformed docs — skipping DB save.", color="yellow")
        return db

    # Step 4: Validate uniform vectors and update blob cache
    valid_docs = []
    invalid_count = 0
    blob_groups: Dict[str, List[Dict[str, Any]]] = {}

    for doc in transformed_docs:
        if hasattr(doc, "vector") and doc.vector and len(doc.vector) == expected_dim:
            valid_docs.append(doc)
            b_hash = (doc.meta_data or {}).get("blob_hash")
            if b_hash:
                if b_hash not in blob_groups:
                    blob_groups[b_hash] = []
                blob_groups[b_hash].append({
                    "text": doc.text,
                    "vector": doc.vector,
                    "meta_data": doc.meta_data,
                })
        else:
            invalid_count += 1

    # Cache embedded chunks for each Git blob hash
    for b_hash, chunk_list in blob_groups.items():
        dedup.cache_chunks(b_hash, chunk_list)

    if invalid_count > 0:
        printc(
            f"DataPipeline: Filtered out {invalid_count} chunks with missing or invalid vectors (expected {expected_dim} dimensions).",
            color="yellow",
        )

    if not valid_docs:
        raise RuntimeError(
            f"Embedding pipeline failed: None of the {len(transformed_docs)} generated chunks received valid "
            f"{expected_dim}-dimensional embeddings. Please verify your GEMINI_API_KEY."
        )

    printc(
        f"DataPipeline: Successfully produced {len(valid_docs)} valid {expected_dim}-dim document embeddings.",
        color="green",
    )

    db.data["split_and_embed"] = valid_docs
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.transformer_setups = {}
    try:
        LocalDB.save_state(db, filepath=db_path)
        printc(f"DataPipeline: Saved database to {db_path}", color="green")
    except Exception as e:
        printc(f"Failed saving DB: {e}", color="red")

    return db


# Database manager
class DatabaseManager:
    def __init__(self):
        self.db = None
        self.repo_paths = None

    def prepare_database(
        self,
        repo_url_or_path: str,
        force_reindex: bool = False,
        token: str | None = None,
    ):
        """Prepare local repository storage and index/load document embeddings."""
        printc(f"Preparing repo storage for {repo_url_or_path}...")
        root_path = get_adalflow_default_root_path()
        os.makedirs(root_path, exist_ok=True)

        meta = extract_repo_metadata(repo_url_or_path)
        slug = get_repo_slug(repo_url_or_path)

        if meta.get("is_local", False):
            save_repo_dir = meta["canonical_url"]
        else:
            save_repo_dir = os.path.join(root_path, "repos", slug)
            if not os.path.exists(save_repo_dir) or not os.listdir(save_repo_dir):
                download_github_repo(repo_url_or_path, save_repo_dir, token=token)

        save_db_file = os.path.join(root_path, "databases", f"{slug}.pkl")
        os.makedirs(save_repo_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_db_file), exist_ok=True)
        self.repo_paths = {"save_repo_dir": save_repo_dir, "save_db_file": save_db_file, "slug": slug}

        expected_dim = config["embedder"].get("dimensions", 768)

        # If cached database exists and reindexing is not forced, validate cached embeddings
        if not force_reindex and os.path.exists(save_db_file) and os.path.getsize(save_db_file) > 0:
            printc("Checking existing cached database...", color="blue")
            try:
                self.db = LocalDB.load_state(save_db_file)
                docs = self.db.get_transformed_data(key="split_and_embed")
                if docs and len(docs) > 0:
                    if all(hasattr(d, "vector") and d.vector and len(d.vector) == expected_dim for d in docs):
                        printc(
                            f"Cache verified: Loaded {len(docs)} documents with uniform {expected_dim}-dim vectors.",
                            color="green",
                        )
                        return docs
                    printc("Corrupted cache detected. Purging cache and re-indexing...", color="yellow")
                    try:
                        os.remove(save_db_file)
                    except OSError as rm_err:
                        log.warning(f"Could not remove corrupted cache file {save_db_file}: {rm_err}")
            except Exception as e:
                printc(f"Failed to load cached DB ({e}) — reindexing.", color="yellow")
                try:
                    if os.path.exists(save_db_file):
                        os.remove(save_db_file)
                except OSError as rm_err:
                    log.warning(f"Could not remove invalid cache file {save_db_file}: {rm_err}")

        printc("Indexing repository...", color="green")
        docs = stream_and_index_repository_batches(
            save_repo_dir=save_repo_dir,
            save_db_file=save_db_file,
            repo_slug=slug,
            batch_file_count=150,
        )
        if docs:
            return docs

        # Fallback to read_all_documents if streaming produced 0 files
        docs = read_all_documents(save_repo_dir)
        if not docs:
            raise ValueError(f"No readable code or text files found in {save_repo_dir}")
        self.db = transform_documents_and_save_to_db(docs, save_db_file)
        return self.db.get_transformed_data(key="split_and_embed")


def stream_and_index_repository_batches(
    save_repo_dir: str,
    save_db_file: str,
    repo_slug: str,
    batch_file_count: int = 150,
) -> List[Document]:
    """Stream valid code files in fixed batches, generating AST chunks and embeddings with constant O(1) RAM."""
    redis_mgr = RedisCacheManager.get_instance()
    checkpoint = redis_mgr.get_ingest_checkpoint(repo_slug) if redis_mgr else None
    start_index = checkpoint.get("last_processed_index", 0) if checkpoint else 0

    accumulated_embedded_docs: List[Document] = []
    current_batch_docs: List[Document] = []
    file_counter = 0

    log.info(f"Streaming ingestion started for {save_repo_dir} (checkpoint resume index: {start_index})")

    for doc in stream_documents(save_repo_dir):
        file_counter += 1
        if file_counter <= start_index:
            continue

        current_batch_docs.append(doc)

        if len(current_batch_docs) >= batch_file_count:
            try:
                batch_db = transform_documents_and_save_to_db(current_batch_docs, save_db_file)
                batch_embedded = batch_db.get_transformed_data(key="split_and_embed") or []
                accumulated_embedded_docs.extend(batch_embedded)

                if redis_mgr:
                    redis_mgr.set_ingest_checkpoint(repo_slug, file_counter, total_files=-1)
            except Exception as batch_err:
                log.warning(f"Error processing streaming batch at file #{file_counter}: {batch_err}")

            current_batch_docs.clear()
            gc.collect()

    if current_batch_docs:
        try:
            batch_db = transform_documents_and_save_to_db(current_batch_docs, save_db_file)
            batch_embedded = batch_db.get_transformed_data(key="split_and_embed") or []
            accumulated_embedded_docs.extend(batch_embedded)
        except Exception as batch_err:
            log.warning(f"Error processing final batch at file #{file_counter}: {batch_err}")
        current_batch_docs.clear()
        gc.collect()

    if redis_mgr:
        redis_mgr.clear_ingest_checkpoint(repo_slug)

    return accumulated_embedded_docs


def sync_git_delta(
    repo_url: str,
    local_path: str,
    previous_commit: str,
    new_commit: str = "HEAD",
    qdrant_manager: Optional[Any] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Incremental Change Data Capture (CDC): identifies changed, added, and deleted files between Git commits

    and synchronizes Qdrant vector store by removing deleted/modified file points and returning
    only newly added/modified documents for AST chunking and embedding.
    """
    try:
        cmd = ["git", "diff", "--name-status", previous_commit, new_commit]
        res = subprocess.run(
            cmd,
            cwd=local_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        diff_lines = res.stdout.strip().splitlines()
    except Exception as e:
        log.warning(f"Git diff failed between {previous_commit} and {new_commit}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "deleted_files": [],
            "modified_files": [],
            "added_files": [],
            "updated_docs": [],
        }

    deleted_files: List[str] = []
    modified_files: List[str] = []
    added_files: List[str] = []

    for line in diff_lines:
        parts = line.strip().split("\t")
        if not parts or len(parts) < 2:
            continue
        status_code = parts[0][0].upper()
        if status_code == "D":
            deleted_files.append(parts[1])
        elif status_code == "A":
            added_files.append(parts[1])
        elif status_code == "M":
            modified_files.append(parts[1])
        elif status_code == "R":
            deleted_files.append(parts[1])
            if len(parts) > 2:
                added_files.append(parts[2])

    # Delete points for deleted and modified files from Qdrant
    files_to_remove = set(deleted_files + modified_files)
    if qdrant_manager and collection_name:
        for fpath in files_to_remove:
            normalized_rel = fpath.replace("\\", "/")
            try:
                qdrant_manager.delete_points_by_file_path(collection_name, normalized_rel)
            except Exception as del_err:
                log.warning(f"Error deleting points for {fpath}: {del_err}")

    # Read and construct Document objects for added + modified files
    updated_docs: List[Document] = []
    files_to_read = set(added_files + modified_files)
    for fpath in sorted(list(files_to_read)):
        full_path = os.path.join(local_path, fpath)
        normalized_rel = fpath.replace("\\", "/")
        _, ext = os.path.splitext(fpath)
        ext_clean = ext.lower()
        if ext_clean not in ALL_INDEXABLE_EXTENSIONS:
            continue
        if not os.path.exists(full_path):
            continue
        try:
            size = os.path.getsize(full_path)
            if size == 0 or size > MAX_FILE_SIZE_BYTES:
                continue
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content.strip()) < MIN_FILE_CHAR_LENGTH:
                continue
            is_code = ext_clean in CODE_EXTENSIONS
            updated_docs.append(
                Document(
                    text=content,
                    meta_data={
                        "file_path": normalized_rel,
                        "type": ext_clean.lstrip("."),
                        "is_code": is_code,
                        "is_implementation": is_code,
                        "title": normalized_rel,
                        "repo_url": repo_url,
                    },
                )
            )
        except Exception as read_err:
            log.warning(f"Failed to read modified file {fpath}: {read_err}")

    log.info(
        f"Git Delta CDC: {len(deleted_files)} deleted, {len(modified_files)} modified, "
        f"{len(added_files)} added -> {len(updated_docs)} docs to re-index"
    )
    return {
        "status": "success",
        "deleted_files": deleted_files,
        "modified_files": modified_files,
        "added_files": added_files,
        "updated_docs": updated_docs,
    }
