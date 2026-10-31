"""Compare browser and Python rankings using the committed serving artifact."""
import json,subprocess,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving import load_artifact,recommend
data=load_artifact();products=data['products'];fixtures=[]
for liked,excluded,limit,category,query in [([],[],6,'',''),([products[0]['id']],[],5,'',''),([p['id'] for p in products[10:15]],[products[0]['id']],20,'',''),([],[],12,products[0]['category'],''),([],[],6,'','does-not-exist'),([],[],20,'','clean')]:
    fixtures.append(dict(liked=liked,excluded=excluded,limit=limit,category=category,query=query,expected=recommend(data,liked,excluded,limit,category,query)))
with tempfile.TemporaryDirectory() as tmp:
    fixture=Path(tmp)/'fixtures.json';fixture.write_text(json.dumps(fixtures),encoding='utf-8')
    subprocess.run(['node',str(ROOT/'tests/browser-parity.mjs'),str(fixture)],check=True)
