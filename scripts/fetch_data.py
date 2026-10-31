"""Restore the pinned public dataset copies listed in dataset-sources.json."""
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def restore_bundled(entry, target, directory):
    """Reconstruct stored data without contacting any external host."""
    import gzip, shutil
    index_path = ROOT / 'dataset' / 'bundled-index.json'
    if not index_path.exists():
        return False
    record = json.loads(index_path.read_text(encoding='utf-8'))['files'].get(entry['filename'])
    if not record or record['sha256'] != entry['sha256']:
        return False
    base = (ROOT / 'dataset').resolve()
    def safe_path(relative):
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError('Invalid stored dataset path')
        return path
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        payload=Path(temporary)/'payload'
        if record['storage']=='raw':
            original=safe_path(record['path'])
            if not original.exists():return False
            shutil.copyfile(original,payload)
        else:
            archive=Path(temporary)/'data.gz'
            with archive.open('wb') as combined:
                for part in record['parts']:
                    path=safe_path(part['path'])
                    if not path.exists():return False
                    if digest(path)!=part['sha256']:raise ValueError('Stored dataset part checksum mismatch')
                    with path.open('rb') as stream:shutil.copyfileobj(stream,combined)
            with gzip.open(archive,'rb') as stream,payload.open('wb') as out:shutil.copyfileobj(stream,out)
        if digest(payload)!=entry['sha256']:raise ValueError('Stored dataset payload checksum mismatch')
        payload.replace(target)
    return True

def ensure_data(directory=None, allow_network=None):
    directory = Path(directory or os.environ.get("DATA_DIR", ROOT / "dataset"))
    directory.mkdir(parents=True, exist_ok=True)
    if allow_network is None:
        allow_network = os.environ.get("DATA_OFFLINE") != "1"
    manifest = json.loads((ROOT / "dataset-sources.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if Path(entry["filename"]).name != entry["filename"]:
            raise ValueError("Manifest filenames must be plain filenames")
        target = directory / entry["filename"]
        if target.exists():
            if digest(target) not in [entry["sha256"], *entry.get("compatible_sha256", [])]:
                raise ValueError(f"{target} differs from the pinned copy. Move it aside to restore the reference dataset.")
            continue
        if restore_bundled(entry, target, directory):
            continue
        if not allow_network:
            raise FileNotFoundError("Stored reference data is missing for " + entry["filename"])
        print(f"Downloading {entry['filename']} from the recorded public source...", flush=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            download = Path(temporary) / "download"
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "reproducible-dataset-recovery"})
            with urllib.request.urlopen(request, timeout=180) as response, download.open("wb") as stream:
                import shutil
                shutil.copyfileobj(response, stream)
            if entry.get("archive_member"):
                payload = Path(temporary) / "payload"
                with zipfile.ZipFile(download) as archive, archive.open(entry["archive_member"]) as source, payload.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
            else:
                payload = download
            if digest(payload) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for {entry['filename']}; download was not installed.")
            payload.replace(target)
    return directory

def ensure_nltk():
    import nltk
    cache = ROOT / ".cache" / "nltk_data"
    cache.mkdir(parents=True, exist_ok=True)
    if str(cache) not in nltk.data.path:
        nltk.data.path.insert(0, str(cache))
    resources = {"punkt_tab": "tokenizers/punkt_tab", "wordnet": "corpora/wordnet.zip",
                 "stopwords": "corpora/stopwords", "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"}
    for package, resource in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=str(cache), quiet=True, raise_on_error=True)

if __name__ == "__main__":
    ensure_data()
    print("Dataset files restored and SHA-256 verified.")
