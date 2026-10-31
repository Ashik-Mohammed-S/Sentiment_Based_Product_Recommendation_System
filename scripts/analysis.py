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
    from scripts.recommender import evaluate
    recommendation = evaluate()
    report['recommendation_evaluation'] = recommendation
    report['rating_model'] = 'Validation-selected shrinkage item means and regularized user bias'
    report['ranking_model'] = 'Validation-selected popularity, smoothed sentiment and EASE item co-occurrence'
    return finish(report,summary)
