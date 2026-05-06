"""
Fetch and cache MovieLens dataset.
"""

import zipfile
import urllib.request
from pathlib import Path

MOVIELENS_URLS = {
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}

DEFAULT_VERSION = "ml-32m"


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parents[3]


def get_dataset_dir() -> Path:
    """Get the .dataset directory path."""
    return get_repo_root() / ".dataset"


def fetch_movielens(version: str = DEFAULT_VERSION, force: bool = False) -> Path:
    """
    Fetch MovieLens dataset if not already cached.

    Args:
        version: Dataset version (ml-32m)
        force: Force re-download even if cached

    Returns:
        Path to the extracted dataset directory
    """
    if version not in MOVIELENS_URLS:
        raise ValueError(
            f"Unknown version: {version}. Available: {list(MOVIELENS_URLS.keys())}"
        )

    dataset_dir = get_dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    extracted_path = dataset_dir / version

    if extracted_path.exists() and not force:
        print(f"Dataset already cached at {extracted_path}")
        return extracted_path

    url = MOVIELENS_URLS[version]
    zip_path = dataset_dir / f"{version}.zip"

    print(f"Downloading {version} from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting to {dataset_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    zip_path.unlink()
    print(f"Dataset ready at {extracted_path}")

    return extracted_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MovieLens dataset")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        choices=list(MOVIELENS_URLS.keys()),
        help="Dataset version",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    fetch_movielens(version=args.version, force=args.force)
