"""Runner-owned checks. A private sign-in gate is reachable, not authenticated app QA."""
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SITE_URL='https://review-compass-ashik.swoonysnipe19.chatgpt.site/'

def check_local_app():
    from app import create_app
    from serving import load_artifact
    model=load_artifact();app=create_app(model)
    with app.test_client() as client:
        with client.get('/health') as health:
            assert health.status_code==200 and health.json['status']=='ok'
            assert health.json['products']==len(model['products'])>0
        with client.post('/api/recommend',json={'liked_ids':[],'limit':6}) as response:
            assert response.status_code==200
            ids=[p['id'] for p in response.json['recommendations']]
            assert len(ids)==len(set(ids))==6
        with client.post('/api/recommend',json={'liked_ids':[ids[0]],'limit':6}) as response:
            assert response.status_code==200
            assert ids[0] not in {p['id'] for p in response.json['recommendations']}
    return 'Flask artifact, local health endpoint and cold/personalized inference passed.'

def probe_site(opener=urlopen):
    request=Request(SITE_URL,headers={'User-Agent':'ReviewCompass-Repository-Monitor/1.0'})
    try:
        response=opener(request,timeout=20)
    except HTTPError as error:
        # Sites returns its expected sign-in HTML with HTTP 401. Only a
        # recognizable gate counts as reachable; arbitrary 401/403 errors do not.
        if error.code not in (401,403):raise
        response=error
    with response:
        status=response.status
        page=response.read(512_000).decode('utf-8',errors='replace').lower()
        gate=('log in to access' in page or 'continue with chatgpt' in page or '<title>sign in required</title>' in page)
        if status in (200,401,403) and gate:
            return 'Private ChatGPT sign-in gate reachable; authenticated hosted inference is not checked.'
        if status!=200:raise RuntimeError('Hosted endpoint returned an unexpected access error')
        if 'review compass' in page and 'product discovery' in page:
            return 'Hosted app HTML reachable; authenticated hosted inference is not checked.'
        raise RuntimeError('Hosted response is neither the app page nor the expected private sign-in gate')

def check_site(attempts=3,probe=probe_site,sleep=time.sleep):
    for attempt in range(attempts):
        try:return probe()
        except Exception as error:
            print(f'Hosted probe attempt {attempt+1} failed: {type(error).__name__}',file=sys.stderr)
            if attempt+1==attempts:raise RuntimeError('Hosted reachability check failed after three attempts') from None
            sleep(10)

if __name__=='__main__':
    print(check_local_app())
    print(check_site())
