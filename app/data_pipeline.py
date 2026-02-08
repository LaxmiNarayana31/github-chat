import os
import glob
import subprocess

import adalflow as adal
from adalflow.utils import printc
from adalflow.core.db import LocalDB
from adalflow.core.types import Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings

from app.config import config

# Clone github repo to local path
def download_github_repo(repo_url: str, local_path: str):
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.makedirs(local_path, exist_ok=True)
        result = subprocess.run(["git", "clone", repo_url, local_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        return f"Error during cloning: {e.stderr.decode('utf-8')}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Read all documents from local path
def read_all_documents(path: str):
    documents = []
    code_exts = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]
    doc_exts = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    for ext in code_exts + doc_exts:
        for file_path in glob.glob(f"{path}/**/*{ext}", recursive=True):
            if ".venv" in file_path or "node_modules" in file_path:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    rel = os.path.relpath(file_path, path)
                    is_code = ext in code_exts
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
                print(f"Error reading {file_path}: {e}")
    return documents

# Prepare data pipeline for embedding
def prepare_data_pipeline():
    splitter = TextSplitter(**config["text_splitter"])
    embedder = adal.Embedder(
        model_client=config["embedder"]["model_client"](),
        model_kwargs=config["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=config["embedder"]["batch_size"]
    )
    return adal.Sequential(splitter, embedder_transformer)

# Transform documents and save to db
def transform_documents_and_save_to_db(documents: List[Document], db_path: str) -> LocalDB:
    transformer = prepare_data_pipeline()
    db = LocalDB()
    db.register_transformer(transformer=transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")

    transformed_docs = db.get_transformed_data(key="split_and_embed")
    if not transformed_docs:
        printc("No embedded docs — skipping DB save.")
        return db

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.transformer_setups = {}
    try:
        LocalDB.save_state(db, filepath=db_path)
    except Exception as e:
        printc(f"Failed saving DB: {e}")
    return db

