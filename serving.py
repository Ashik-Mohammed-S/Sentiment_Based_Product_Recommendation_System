"""Dependency-light inference for the portable, versioned recommendation artifact."""
import json
from pathlib import Path

def load_artifact(path=None):
    path=Path(path) if path else Path(__file__).parent/'web/model.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version')!=1:raise ValueError('Unsupported model artifact')
    return data

def recommend(data, liked_ids=(), excluded_ids=(), limit=6, category='', query=''):
    products=data['products']; indices={p['id']:i for i,p in enumerate(products)}
    liked=list(dict.fromkeys(i for i in liked_ids if i in indices))
    exclude=set(liked)|set(excluded_ids)
    profile=[indices[i] for i in liked];config=data['config']
    collab=[max(0,sum(data['ease'][j][i] for j in profile)) for i in range(len(products))]
    scale=max(collab,default=0) or 1
    ranked=[]
    for i,p in enumerate(products):
        if p['id'] in exclude or (category and p['category']!=category):continue
        if query.casefold() not in (p['name']+' '+p['brand']+' '+p['categories']).casefold():continue
        base=data['popularity'][i]*data['quality'][i]**config['sentiment_power']
        content=sum(data['content'][j][i] for j in profile)/len(profile) if profile else 0
        score=base+config['collaborative_weight']*collab[i]/scale+config['content_weight']*content
        reason='Popular with reviewers'
        if profile and config['collaborative_weight']>0 and collab[i]/scale>.1:reason='Related to your picks'
        elif config['sentiment_power'] and p['positive_fraction']>.9:reason='Strong review sentiment'
        ranked.append({**p,'score':round(score,8),'reason':reason})
    return sorted(ranked,key=lambda p:(-p['score'],p['id']))[:limit]
