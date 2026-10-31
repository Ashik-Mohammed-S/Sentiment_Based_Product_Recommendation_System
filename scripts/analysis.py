from scripts.analysis_common import *
from scripts.fetch_data import ensure_data

def run(mode='full'):
    from sklearn.model_selection import GroupShuffleSplit,train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    from scipy.sparse import csr_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    data=ensure_data(ROOT/'dataset');df=pd.read_csv(data/'sample30.csv')
    df=df.loc[df.user_sentiment.isin(['Positive','Negative'])].copy()
    df['target']=df.user_sentiment.map({'Negative':0,'Positive':1})
    df['text']=df.reviews_text.fillna('').astype(str).str.replace(r'\s+',' ',regex=True).str.strip()
    df=df.loc[df.text.ne('')].copy()
    if mode=='quick':df=df.sample(n=min(4000,len(df)),random_state=42)
    groups=df.text.str.lower()
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,df.target,groups=groups))
    assert not set(groups.iloc[train])&set(groups.iloc[test])
    model=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',min_df=2,max_features=20000,ngram_range=(1,2))),
        ('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
    report,summary,model=classifier_report(model,df.text.iloc[train],df.text.iloc[test],df.target.iloc[train],df.target.iloc[test],
        'Classify review sentiment, then recommend unseen products from training-only collaborative signals.','user_sentiment',
        'Text-group 80/20 holdout: identical normalized reviews cannot cross the split. TF-IDF and class balancing fit only on training data; no pre-CV SMOTE or test vocabulary fitting.')
    # Separate rating-evaluation partition, grouped by user/item pair to avoid repeated-pair leakage.
    ratings=df[['reviews_username','id','reviews_rating']].dropna().groupby(['reviews_username','id'],as_index=False).reviews_rating.mean()
    rating_train,rating_test=train_test_split(ratings,test_size=.2,random_state=42)
    users=sorted(rating_train.reviews_username.unique());items=sorted(rating_train.id.unique())
    ui={v:i for i,v in enumerate(users)};ii={v:i for i,v in enumerate(items)}
    matrix=csr_matrix((rating_train.reviews_rating,(rating_train.reviews_username.map(ui),rating_train.id.map(ii))),shape=(len(users),len(items)))
    similarity=cosine_similarity(matrix.T,dense_output=True);np.fill_diagonal(similarity,0)
    global_mean=float(rating_train.reviews_rating.mean())
    user_mean=rating_train.groupby('reviews_username').reviews_rating.mean().to_dict()
    def predict_rating(user,item):
        if user not in ui or item not in ii:return global_mean,False
        row=matrix.getrow(ui[user]);weights=similarity[row.indices,ii[item]]
        if weights.sum()<=0:return user_mean[user],True
        return float(np.clip(np.dot(weights,row.data)/weights.sum(),1,5)),True
    predictions=[predict_rating(r.reviews_username,r.id) for r in rating_test.itertuples()]
    values=np.array([v[0] for v in predictions]);coverage=np.mean([v[1] for v in predictions])
    report['recommendation_evaluation']={'target':'held-out user/product mean rating','train_pairs':len(rating_train),'test_pairs':len(rating_test),
        'rmse':float(np.sqrt(mean_squared_error(rating_test.reviews_rating,values))),
        'mae':mean_absolute_error(rating_test.reviews_rating,values),
        'global_mean_baseline_rmse':float(np.sqrt(mean_squared_error(rating_test.reviews_rating,np.full(len(values),global_mean)))),
        'warm_start_coverage':float(coverage),
        'interpretation':'All held-out pairs are evaluated. Unknown users/items use the training global mean. Warm-start coverage is reported rather than hiding cold-start cases. No test ratings enter the similarity matrix.'}
    # Evaluate the actual top-K recommendation objective separately from rating RMSE.
    relevant=rating_test.loc[rating_test.reviews_rating>=4].groupby('reviews_username').id.agg(set)
    item_popularity=rating_train.groupby('id').reviews_rating.agg(['mean','size']).reindex(items)
    popularity_order=item_popularity.sort_values(['size','mean'],ascending=False).index.tolist()
    scores=[];baseline_scores=[]
    for user, truth in relevant.items():
        if user not in ui:continue
        row=matrix.getrow(ui[user]);seen_items={items[i] for i in row.indices}
        truth=set(truth)&set(items)-seen_items
        if not truth:continue
        weights=similarity[row.indices,:]
        denom=weights.sum(axis=0)
        values_for_user=np.divide(row.data @ weights,denom,out=np.full(len(items),global_mean),where=denom>0)
        candidate_indices=[i for i,item in enumerate(items) if item not in seen_items]
        top=[items[i] for i in sorted(candidate_indices,key=lambda i:values_for_user[i],reverse=True)[:5]]
        base=[item for item in popularity_order if item not in seen_items][:5]
        scores.append((len(set(top)&truth)/len(top),len(set(top)&truth)/len(truth)))
        baseline_scores.append((len(set(base)&truth)/len(base),len(set(base)&truth)/len(truth)))
    report['ranking_evaluation']={'k':5,'evaluated_warm_users':len(scores),
        'precision_at_5':float(np.mean([r[0] for r in scores])) if scores else None,
        'recall_at_5':float(np.mean([r[1] for r in scores])) if scores else None,
        'popularity_precision_at_5':float(np.mean([r[0] for r in baseline_scores])) if scores else None,
        'popularity_recall_at_5':float(np.mean([r[1] for r in baseline_scores])) if scores else None,
        'interpretation':'Relevant means held-out rating >=4. Evaluated on users/items seen in training with a held-out relevant item. Unobserved items are unjudged; this is an offline proxy, not an online engagement claim.'}
    example=rating_train.reviews_username.value_counts().index[0]
    seen=set(rating_train.loc[rating_train.reviews_username==example,'id']);candidates=[item for item in items if item not in seen]
    ranked=sorted(((item,predict_rating(example,item)[0]) for item in candidates),key=lambda p:p[1],reverse=True)[:5]
    pd.DataFrame(ranked,columns=['product_id','predicted_rating']).to_csv(output()/'example_recommendations.csv',index=False)
    report['recommendation_example_user']=str(example)
    report['rating_model']='Item-item similarity (product-count squared), avoiding a dense user-by-user matrix.'
    return finish(report,summary)
