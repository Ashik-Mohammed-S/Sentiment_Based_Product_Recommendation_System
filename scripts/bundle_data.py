"""Store verified reference data as raw files or bounded gzip parts in this repository."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_data import ensure_data,digest

def bundle():
    directory=ensure_data(ROOT/'dataset',allow_network=True)
    source=json.loads((ROOT/'dataset-sources.json').read_text())
    bundle_dir=directory/'bundled';bundle_dir.mkdir(exist_ok=True)
    index={'version':1,'files':{}};paths=[]
    for entry in source['files']:
        path=directory/entry['filename']
        # Normalize only a recognized Git line-ending variant to its pinned canonical bytes.
        if digest(path)!=entry['sha256']:
            canonical=path.read_bytes().replace(b'\r\n',b'\n')
            if hashlib.sha256(canonical).hexdigest()!=entry['sha256']:
                raise ValueError('Cannot canonicalize '+entry['filename'])
            path.write_bytes(canonical)
        record={'sha256':entry['sha256'],'uncompressed_bytes':path.stat().st_size}
        tracked = subprocess.run(['git','ls-files','--error-unmatch',path.relative_to(ROOT).as_posix()],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
        if path.stat().st_size<=8*1024*1024 or (tracked and path.stat().st_size<50*1024*1024):
            record.update(storage='raw',path=entry['filename'])
            paths.append(path.relative_to(ROOT).as_posix())
        else:
            parts=[]
            with tempfile.TemporaryDirectory(dir=bundle_dir) as temporary:
                archive=Path(temporary)/'data.gz'
                with path.open('rb') as stream,archive.open('wb') as target:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0,compresslevel=6) as compressed:
                        shutil.copyfileobj(stream,compressed)
                with archive.open('rb') as stream:
                    number=0
                    while True:
                        chunk=stream.read(32*1024*1024)
                        if not chunk:break
                        part=bundle_dir/(entry['filename']+f'.gz.part{number:03d}')
                        part.write_bytes(chunk)
                        parts.append({'path':part.relative_to(directory).as_posix(),'sha256':digest(part),'bytes':len(chunk)})
                        paths.append(part.relative_to(ROOT).as_posix());number+=1
            record.update(storage='gzip_parts',parts=parts)
        index['files'][entry['filename']]=record
    index_path=directory/'bundled-index.json'
    index_path.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8');paths.append(index_path.relative_to(ROOT).as_posix())
    print(json.dumps({'stored_files':len(index['files']),'git_paths':paths}),flush=True)
    return paths

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',action='store_true');args=parser.parse_args()
    paths=bundle()
    if args.commit:
        branch='codex/latest-stable-2026-11-01'
        if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REF')!='refs/heads/'+branch:
            raise RuntimeError('Automatic commits are restricted to the authorized preparation branch in GitHub Actions.')
        subprocess.run(['git','config','user.name','github-actions[bot]'],check=True)
        subprocess.run(['git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com'],check=True)
        subprocess.run(['git','add','-f','--',*paths],check=True,cwd=ROOT)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m','Store verified reference datasets for offline access'],check=True,cwd=ROOT)
            subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],check=True,cwd=ROOT)
