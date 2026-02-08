import os
import subprocess

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


