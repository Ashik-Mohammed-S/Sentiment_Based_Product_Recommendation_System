"""Train-only recommendation fitting, validation selection and portable serving artifact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    from scripts.fetch_data import ensure_data
    df = pd.read_csv(ensure_data(ROOT / 'dataset') / 'sample30.csv')
    df = df.loc[df.user_sentiment.isin(['Positive', 'Negative'])].copy()
    df = df.loc[df.reviews_text.fillna('').astype(str).str.strip().ne('')]
    df['positive'] = (df.user_sentiment == 'Positive').astype(int)
    pairs = df.dropna(subset=['reviews_username', 'id', 'reviews_rating']).groupby(
        ['reviews_username', 'id'], as_index=False).agg(rating=('reviews_rating', 'mean'),
        positive=('positive', 'mean'))
    return df, pairs

def fit(train, metadata, regularization=100):
    items = sorted(train.id.unique())
    users = sorted(train.reviews_username.unique())
    ii = {x:i for i,x in enumerate(items)}
    ui = {x:i for i,x in enumerate(users)}
    x = np.zeros((len(users),len(items)), dtype=np.float64)
    for r in train.itertuples():
        if r.rating >= 4: x[ui[r.reviews_username],ii[r.id]] = 1
    gram = x.T @ x
    inverse = np.linalg.inv(gram + regularization * np.eye(len(items)))
    ease = -inverse / np.diag(inverse)[None,:]
    np.fill_diagonal(ease, 0)
    stats = train.groupby('id').agg(count=('rating','size'), total=('rating','sum'),
                                   positives=('positive','sum')).reindex(items)
    mean = float(train.rating.mean())
    prior = float(train.positive.mean())
    # Regularized quality estimates prevent a single 5-star review winning the ranking.
    quality = (stats.positives.to_numpy()+20*prior)/(stats['count'].to_numpy()+20)
    popularity = stats['count'].to_numpy(dtype=float)
    popularity /= popularity.max()
    meta = metadata.drop_duplicates('id').set_index('id').reindex(items)
    text = (meta.categories.fillna('')+' '+meta.brand.fillna('')+' '+meta['name'].fillna(''))
    content = cosine_similarity(TfidfVectorizer(stop_words='english',max_features=5000).fit_transform(text))
    np.fill_diagonal(content,0)
    # The same compact scoring representation is used by Python, Flask and the browser.
    return dict(items=items, ii=ii, users=ui, ease=ease, content=content,
                popularity=popularity, quality=quality, stats=stats, global_mean=mean,
                user_bias=(train.groupby('reviews_username').rating.agg(['sum','size'])),
                training_users=set(users))

def scores(model, liked, config):
    indices=[model['ii'][i] for i in liked if i in model['ii']]
    base = model['popularity'] * (model['quality'] ** config['sentiment_power'])
    if not indices: return base
    collaborative = model['ease'][indices].sum(axis=0)
    collaborative = np.maximum(collaborative,0)
    if collaborative.max()>0: collaborative /= collaborative.max()
    content = model['content'][indices].mean(axis=0)
    return base + config['collaborative_weight']*collaborative + config['content_weight']*content

def rank_metrics(model, train, test, config, k=5):
    profiles=train.groupby('reviews_username').apply(
        lambda g: (set(g.id),set(g.loc[g.rating>=4,'id'])),include_groups=False).to_dict()
    truth=test.loc[test.rating>=4].groupby('reviews_username').id.agg(set)
    precision=[]; recall=[]; ndcg=[]; recommended=set(); cold=0
    for user, relevant in truth.items():
        seen,liked=profiles.get(user,(set(),set()))
        if user not in profiles:cold+=1
        # Include cold users and unknown test items in the denominator; no hidden filtering.
        relevant=set(relevant)-seen
        if not relevant:continue
        values=scores(model,liked,config)
        candidates=[i for i,item in enumerate(model['items']) if item not in seen]
        order=sorted(candidates,key=lambda i:(-values[i],model['items'][i]))[:k]
        top=[model['items'][i] for i in order];recommended.update(top)
        hits=np.array([item in relevant for item in top],dtype=float)
        precision.append(float(hits.sum()/k));recall.append(float(hits.sum()/len(relevant)))
        discount=1/np.log2(np.arange(2,len(top)+2))
        ideal=sum(1/np.log2(np.arange(2,min(k,len(relevant))+2)))
        ndcg.append(float((hits*discount).sum()/ideal))
    return dict(precision_at_5=float(np.mean(precision)),recall_at_5=float(np.mean(recall)),
                ndcg_at_5=float(np.mean(ndcg)),evaluated_users=len(precision),cold_users=cold,
                catalog_coverage=len(recommended)/len(model['items']))

def predict_rating(model, user, item, shrink):
    if item not in model['ii']:return model['global_mean']
    s=model['stats'].loc[item]
    item_mean=(s.total+shrink*model['global_mean'])/(s['count']+shrink)
    if user in model['user_bias'].index:
        u=model['user_bias'].loc[user]
        bias=(u['sum']-u['size']*model['global_mean'])/(u['size']+20)
    else:bias=0
    return float(np.clip(item_mean+bias,1,5))

def rating_metrics(model,test,shrink):
    y=test.rating.to_numpy();p=np.array([predict_rating(model,r.reviews_username,r.id,shrink) for r in test.itertuples()])
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),mae=float(np.mean(abs(y-p))),
                global_mean_baseline_rmse=float(np.sqrt(np.mean((y-model['global_mean'])**2))))

def evaluate():
    df,pairs=load_data()
    outer_train,test=train_test_split(pairs,test_size=.2,random_state=42)
    train,validation=train_test_split(outer_train,test_size=.2,random_state=43)
    # Product metadata is static and public; no held-out ratings/sentiment enter fitting.
    candidates=[]
    for regularization in [50,200,1000]:
        model=fit(train,df,regularization)
        for cw,tw,sp in [(0,0,0),(0,0,1),(.05,0,1),(.1,0,1),(.25,0,1),(.5,0,1),(1,0,1),(2,0,1),(0,.1,1),(.25,.1,1)]:
            config=dict(regularization=regularization,collaborative_weight=cw,content_weight=tw,sentiment_power=sp)
            metrics=rank_metrics(model,train,validation,config)
            candidates.append(dict(config=config,metrics=metrics))
    # Deterministic ties prefer simpler models in the declared candidate order.
    selected=max(candidates,key=lambda c:c['metrics']['ndcg_at_5'])
    model=fit(train,df,selected['config']['regularization'])
    shrink_candidates=[(s,rating_metrics(model,validation,s)) for s in [5,20,50,100,500]]
    shrink=min(shrink_candidates,key=lambda x:x[1]['rmse'])[0]
    final=fit(outer_train,df,selected['config']['regularization'])
    baseline=dict(regularization=50,collaborative_weight=0,content_weight=0,sentiment_power=0)
    report=dict(objective='Recommend unseen products, judged by held-out positive interactions.',
        split='User/item pairs grouped before fixed 80/20 final split (seed 42); inner 80/20 validation (seed 43) selects ranking configuration by NDCG@5 and rating shrinkage by RMSE. Test is evaluated once after selection.',
        train_pairs=len(train),validation_pairs=len(validation),final_test_pairs=len(test),
        selected_config=selected['config'],validation=selected['metrics'],
        ranking=rank_metrics(final,outer_train,test,selected['config']),
        popularity_baseline=rank_metrics(final,outer_train,test,baseline),
        rating=rating_metrics(final,test,shrink),rating_shrinkage=shrink,
        candidates=candidates,
        limitations='Random pair holdout, not chronological. Relevant means observed rating >=4; unobserved products are unjudged. Cold users and unseen relevant items remain in evaluation. Offline results are not causal or online engagement evidence.')
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/recommendation-evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def export_artifact(report):
    df,pairs=load_data(); config=report['selected_config'];model=fit(pairs,df,config['regularization'])
    meta=df.drop_duplicates('id').set_index('id')
    products=[]
    for i,item in enumerate(model['items']):
        row=meta.loc[item];s=model['stats'].loc[item]
        products.append(dict(id=item,name=str(row['name']),brand=str(row.brand) if pd.notna(row.brand) else '',
            category=str(row.categories).split(',')[0],categories=str(row.categories),
            reviews=int(s['count']),rating=round(float((s.total+report['rating_shrinkage']*model['global_mean'])/(s['count']+report['rating_shrinkage'])),3),
            positive_fraction=round(float(model['quality'][i]),4)))
    # No usernames, raw review texts or user histories are shipped to the web app.
    artifact=dict(schema_version=1,config=config,products=products,
        popularity=np.round(model['popularity'],8).tolist(),
        quality=np.round(model['quality'],8).tolist(),
        ease=np.round(model['ease'],8).tolist(),content=np.round(model['content'],8).tolist(),
        evaluation={k:v for k,v in report.items() if k!='candidates'})
    dest=ROOT/'web/model.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(artifact,separators=(',',':'),allow_nan=False),encoding='utf-8')
    return artifact

if __name__=='__main__':
    report=evaluate();export_artifact(report)
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))
