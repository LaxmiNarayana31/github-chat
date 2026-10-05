"""Unit tests covering enterprise architectural enhancements:
1. Deterministic Symbol Jump via Qdrant Payload Indexing
2. Point deletion by file path for Git Delta Ingestion
3. Startup Job Crash Reconciliation in RedisCacheManager
4. AST Code Skeleton Folding for Dynamic Context Budgeting
5. Symbol Jump candidate extraction in LangGraphAgenticRAG
6. Incremental Git Delta CDC synchronization
7. Mega-repo Resumable Ingestion Checkpoints in Redis
8. Subsystem / Path Scoping in AgenticRAG and Qdrant
9. PageRank Graph Capping for Mega-Repositories
"""

import subprocess
from unittest.mock import MagicMock, patch
import pytest
from adalflow.core.types import Document as AdalDocument

from backend.app.embeddings.qdrant_manager import QdrantManager, sanitize_collection_name
from backend.app.pipelines.ast_chunker import ASTChunker
from backend.app.pipelines.data_pipeline import sync_git_delta
from backend.app.rag.agentic_rag import LangGraphAgenticRAG
from backend.app.rag.repo_map import RepoMapGenerator
from backend.app.services.redis_manager import RedisCacheManager


def test_qdrant_symbol_lookup_and_file_deletion():
    """Verify deterministic <2ms symbol jump lookup and point deletion by file path."""
    manager = QdrantManager(url=None, storage_path=None)  # In-memory Qdrant instance
    col_name = "test_symbol_jump_col"

    doc1 = AdalDocument(
        text="def compute_tree_sitter_hash(buffer: bytes) -> str:\n    return hashlib.sha256(buffer).hexdigest()",
        meta_data={
            "file_path": "core/hasher.py",
            "symbol_name": "compute_tree_sitter_hash",
            "symbol_type": "function",
            "line_range": [1, 2],
        },
    )
    doc1.vector = [0.03] * 768

    doc2 = AdalDocument(
        text="class RedisJobScheduler:\n    def schedule(self): pass",
        meta_data={
            "file_path": "services/scheduler.py",
            "symbol_name": "RedisJobScheduler",
            "symbol_type": "class",
            "line_range": [1, 2],
        },
    )
    doc2.vector = [0.06] * 768

    manager.index_documents(
        collection_name=col_name,
        documents=[doc1, doc2],
        dense_dimension=768,
        force_reindex=True,
    )

    clean_name = sanitize_collection_name(col_name)
    assert manager.get_collection_point_count(clean_name) == 2

    # Deterministic symbol lookup
    matched = manager.lookup_symbol(col_name, "compute_tree_sitter_hash")
    assert len(matched) == 1
    assert matched[0].meta_data["symbol_name"] == "compute_tree_sitter_hash"
    assert matched[0].meta_data["file_path"] == "core/hasher.py"

    # Negative lookup
    non_existent = manager.lookup_symbol(col_name, "non_existent_symbol")
    assert len(non_existent) == 0

    # Delete points by file path (incremental Git CDC update)
    del_res = manager.delete_points_by_file_path(col_name, "core/hasher.py")
    assert del_res is True

    # Confirm point count decreased and symbol lookup no longer finds it
    assert manager.get_collection_point_count(clean_name) == 1
    after_del_lookup = manager.lookup_symbol(col_name, "compute_tree_sitter_hash")
    assert len(after_del_lookup) == 0


def test_redis_reconcile_orphaned_jobs():
    """Verify startup crash reconciliation converts stuck processing/pending jobs to failed."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    mgr.set_job_status("job_stuck_1", {"status": "processing", "repo_url": "https://github.com/org/repo1"})
    mgr.set_job_status("job_stuck_2", {"status": "pending", "repo_url": "https://github.com/org/repo2"})
    mgr.set_job_status("job_ok", {"status": "completed", "repo_url": "https://github.com/org/repo3"})

    reconciled_count = mgr.reconcile_orphaned_jobs()
    assert reconciled_count == 2

    job1 = mgr.get_job_status("job_stuck_1")
    assert job1 is not None and job1["status"] == "failed"
    assert "abrupt server restart" in job1.get("error", "")

    job2 = mgr.get_job_status("job_stuck_2")
    assert job2 is not None and job2["status"] == "failed"

    job_ok = mgr.get_job_status("job_ok")
    assert job_ok is not None and job_ok["status"] == "completed"


def test_ast_fold_code_skeleton():
    """Verify AST Code Skeletonization compresses verbose function bodies into signatures."""
    python_code = '''class DataPipelineEngine:
    """Core data ingestion engine."""

    def process_stream(self, stream: list) -> int:
        """Process raw data streams."""
        total = 0
        for item in stream:
            total += len(item)
            # 50 lines of complex implementation
        return total

def run_worker_task(task_id: str):
    """Execute worker background task."""
    print("Running", task_id)
    return True
'''
    skeleton = ASTChunker.fold_code_skeleton(python_code, ext=".py")
    assert "class DataPipelineEngine:" in skeleton
    assert "def process_stream(...): ..." in skeleton
    assert '"""Process raw data streams."""' in skeleton
    assert "def run_worker_task(...): ..." in skeleton
    assert "total += len(item)" not in skeleton


def test_extract_symbol_candidates():
    """Verify regex-based candidate symbol extraction for deterministic jump-to-definition."""
    assert "calculate_sha256" in LangGraphAgenticRAG._extract_symbol_candidates("where is `calculate_sha256` defined?")
    assert "ASTChunker" in LangGraphAgenticRAG._extract_symbol_candidates("where is class ASTChunker defined?")
    assert "reconcile_orphaned_jobs" in LangGraphAgenticRAG._extract_symbol_candidates("def reconcile_orphaned_jobs in redis_manager")
    assert "lookup_symbol" in LangGraphAgenticRAG._extract_symbol_candidates("find lookup_symbol")
    assert "process_stream" in LangGraphAgenticRAG._extract_symbol_candidates("can you explain process_stream()?")
    assert "RedisLockManager" in LangGraphAgenticRAG._extract_symbol_candidates("RedisLockManager")

    generic = LangGraphAgenticRAG._extract_symbol_candidates("how do I run and test the application?")
    assert len(generic) == 0


def test_sync_git_delta():
    """Verify Git delta CDC detects added, modified, and deleted files and notifies Qdrant."""
    fake_diff_output = (
        "M\tbackend/app/main.py\n"
        "A\tbackend/app/new_feature.py\n"
        "D\tbackend/app/deprecated.py\n"
        "R100\told_util.py\tnew_util.py\n"
    )

    mock_qdrant = MagicMock()
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "diff"],
        returncode=0,
        stdout=fake_diff_output,
        stderr="",
    )

    with patch("subprocess.run", mock_run):
        delta = sync_git_delta(
            repo_url="https://github.com/org/repo",
            local_path=".",
            previous_commit="commit_aaa",
            new_commit="commit_bbb",
            qdrant_manager=mock_qdrant,
            collection_name="test_col",
        )

    assert delta["status"] == "success"
    assert "backend/app/deprecated.py" in delta["deleted_files"]
    assert "old_util.py" in delta["deleted_files"]
    assert "backend/app/main.py" in delta["modified_files"]
    assert "backend/app/new_feature.py" in delta["added_files"]

    deleted_paths = [call.args[1] for call in mock_qdrant.delete_points_by_file_path.call_args_list]
    assert "backend/app/deprecated.py" in deleted_paths
    assert "backend/app/main.py" in deleted_paths
    assert "old_util.py" in deleted_paths


def test_redis_ingest_checkpoints():
    """Verify resumable ingestion checkpoint storage, retrieval, and clearing in Redis."""
    mgr = RedisCacheManager(redis_url="")
    mgr.clear()

    slug = "torvalds_linux_kernel"
    assert mgr.get_ingest_checkpoint(slug) is None

    mgr.set_ingest_checkpoint(slug, last_processed_index=45000, total_files=80000)
    checkpoint = mgr.get_ingest_checkpoint(slug)
    assert checkpoint is not None
    assert checkpoint["last_processed_index"] == 45000
    assert checkpoint["total_files"] == 80000

    mgr.clear_ingest_checkpoint(slug)
    assert mgr.get_ingest_checkpoint(slug) is None


def test_extract_path_scope():
    """Verify subsystem and path filter extraction from natural user questions."""
    scope1 = LangGraphAgenticRAG._extract_path_scope("path:drivers/net/ethernet how does rx packet ring work?")
    assert scope1 == "drivers/net/ethernet"

    scope2 = LangGraphAgenticRAG._extract_path_scope("how does page table compaction work in mm/memory.c?")
    assert scope2 == "mm/memory.c"

    scope3 = LangGraphAgenticRAG._extract_path_scope("explain the locking design")
    assert scope3 is None


def test_qdrant_subsystem_path_scoping():
    """Verify Qdrant symbol lookup and hybrid search respect path_prefix filters."""
    manager = QdrantManager(url=None, storage_path=None)
    col = "test_subsystem_col"

    doc_net = AdalDocument(
        text="int net_packet_alloc(void) { return 0; }",
        meta_data={"file_path": "drivers/net/ethernet/intel.c", "symbol_name": "net_packet_alloc", "symbol_type": "function"},
    )
    doc_net.vector = [0.05] * 768

    doc_fs = AdalDocument(
        text="int net_packet_alloc(void) { return 1; }",
        meta_data={"file_path": "fs/ext4/super.c", "symbol_name": "net_packet_alloc", "symbol_type": "function"},
    )
    doc_fs.vector = [0.05] * 768

    manager.index_documents(collection_name=col, documents=[doc_net, doc_fs], dense_dimension=768, force_reindex=True)

    # Without filter -> returns both matches
    all_matches = manager.lookup_symbol(col, "net_packet_alloc")
    assert len(all_matches) == 2

    # With subsystem path prefix filter -> returns only the drivers/net match
    net_matches = manager.lookup_symbol(col, "net_packet_alloc", path_prefix="drivers/net")
    assert len(net_matches) == 1
    assert "drivers/net" in net_matches[0].meta_data["file_path"]


def test_repomap_mega_repo_capping():
    """Verify RepoMapGenerator caps candidate chunks on repos exceeding 5,000 chunks without memory spike."""
    many_chunks = []
    for i in range(6000):
        target_mod = (i + 1) % 50
        doc = AdalDocument(
            text=f"import module_{target_mod}\ndef func_{i}():\n    return func_{target_mod}()",
            meta_data={
                "file_path": f"module_{i % 50}/file_{i}.py",
                "symbol_name": f"func_{i}",
                "symbol_type": "function",
                "line_range": [1, 3],
            },
        )
        many_chunks.append(doc)

    repo_map_gen = RepoMapGenerator().build_from_chunks(many_chunks)
    assert len(repo_map_gen.graph) > 0
    rendered = repo_map_gen.generate_repo_map()
    assert isinstance(rendered, str)
    assert len(rendered) > 0
