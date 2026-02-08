import os
import glob
import subprocess

from adalflow.core.types import Document


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
