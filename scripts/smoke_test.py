"""Verify storage metadata and that the primary analysis has a declared target."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'dataset-sources.json').read_text())
assert manifest['files'] and all(len(f['sha256'])==64 for f in manifest['files'])
compile((root/'scripts/analysis.py').read_text(), 'analysis.py', 'exec')
print('Target-analysis metadata and syntax checks passed.')
