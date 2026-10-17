from huggingface_hub import HfApi, snapshot_download

from common import MODELS, PARENT_DIR, PARENT_REPO, PRISM_DIR, PRISM_REPO, write_json

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "NOTICE*"]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revs = {}
    for repo, target in ((PRISM_REPO, PRISM_DIR), (PARENT_REPO, PARENT_DIR)):
        sha = api.model_info(repo).sha
        # pin the exact commit so a later rerun reads identical bytes
        snapshot_download(repo, revision=sha, local_dir=target, allow_patterns=PATTERNS, max_workers=2)
        revs[repo] = sha
        print(repo, sha, flush=True)
    write_json(MODELS / "revisions.json", revs)


if __name__ == "__main__":
    main()
