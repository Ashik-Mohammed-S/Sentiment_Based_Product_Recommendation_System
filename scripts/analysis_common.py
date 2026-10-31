"""Shared reporting helpers for target-aware, held-out analyses."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def output():
    path = ROOT / 'results'
    path.mkdir(exist_ok=True)
    return path

def finish(report, summary):
    def convert(value):
        if isinstance(value, dict): return {str(k): convert(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [convert(v) for v in value]
        if isinstance(value, (np.integer, np.floating)): value = value.item()
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    for key, value in report.get('metrics', {}).items():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError('Non-finite analysis metric: ' + key)
    report = convert(report)
    output().joinpath('analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    summary.to_csv(output()/'target_summary.csv', index=False)
    return report

def classification_metrics(y, predictions, probabilities, classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
        average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss)
    result = {'accuracy': accuracy_score(y,predictions), 'balanced_accuracy': balanced_accuracy_score(y,predictions),
              'macro_f1': f1_score(y,predictions,average='macro',zero_division=0)}
    if len(classes)==2:
        positive=classes[1]; truth=np.asarray(y)==positive; prob=probabilities[:,1]
        result.update(precision=precision_score(y,predictions,pos_label=positive,zero_division=0),
            recall=recall_score(y,predictions,pos_label=positive,zero_division=0),
            f1=f1_score(y,predictions,pos_label=positive,zero_division=0),
            roc_auc=roc_auc_score(truth,prob), average_precision=average_precision_score(truth,prob),
            brier_score=brier_score_loss(truth,prob))
    return result

def classifier_report(model, X_train, X_test, y_train, y_test, objective, target, split_note):
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    model.fit(X_train,y_train)
    pred=model.predict(X_test); prob=model.predict_proba(X_test)
    baseline=DummyClassifier(strategy='prior').fit(np.zeros((len(y_train),1)),y_train)
    bp=baseline.predict(np.zeros((len(y_test),1))); bprob=baseline.predict_proba(np.zeros((len(y_test),1)))
    report={'objective':objective,'analysis_type':'held-out classification','target':target,
        'split':split_note,'train_rows':len(y_train),'test_rows':len(y_test),
        'metrics':classification_metrics(y_test,pred,prob,model.classes_),
        'baseline':classification_metrics(y_test,bp,bprob,baseline.classes_),
        'class_report':classification_report(y_test,pred,output_dict=True,zero_division=0)}
    fig,ax=plt.subplots(figsize=(max(6,len(model.classes_)*.65),max(5,len(model.classes_)*.65)))
    ConfusionMatrixDisplay.from_predictions(y_test,pred,normalize='true',xticks_rotation=90,ax=ax,colorbar=False)
    ax.set_title('Held-out target classes: row-normalized confusion matrix')
    fig.tight_layout();fig.savefig(output()/'overview.png',dpi=100);plt.close(fig)
    summary=pd.DataFrame({'target':list(model.classes_),'train_count':[sum(np.asarray(y_train)==c) for c in model.classes_],
        'test_count':[sum(np.asarray(y_test)==c) for c in model.classes_]})
    return report,summary,model

def tabular_classifier(df,target,features,objective,categorical=()):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    df=df.loc[df[target].notna()].copy()
    X=df[features].copy();y=df[target].astype(int)
    cats=[c for c in features if c in categorical or not pd.api.types.is_numeric_dtype(X[c])]
    nums=[c for c in features if c not in cats]
    for c in cats:X[c]=X[c].fillna('Missing').astype(str)
    for c in nums:X[c]=pd.to_numeric(X[c],errors='coerce')
    train,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
    preprocess=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
        ('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=3),cats)])
    model=Pipeline([('preprocess',preprocess),('classifier',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))])
    report,summary,_=classifier_report(model,X.iloc[train],X.iloc[test],y.iloc[train],y.iloc[test],objective,target,
        'Stratified 80/20 holdout; imputers, scaling and category encoding fit only on training rows. Fixed 0.5 binary threshold; no test-set threshold tuning.')
    report['features']=features
    return finish(report,summary)

def temporal_regression(frame,features,target,objective):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    import matplotlib.pyplot as plt
    data=frame.dropna(subset=features+[target]).sort_index()
    cut=int(len(data)*.8);train=data.iloc[:cut];test=data.iloc[cut:]
    assert train.index.max()<test.index.min()
    pipe=Pipeline([('scale',StandardScaler()),('model',Ridge())])
    search=GridSearchCV(pipe,{'model__alpha':[.1,1.,10.,100.]},cv=TimeSeriesSplit(n_splits=3),scoring='neg_mean_absolute_error')
    search.fit(train[features],train[target]);pred=search.predict(test[features]);naive=test['lag_1'].to_numpy()
    score=lambda p:{'mae':mean_absolute_error(test[target],p),'rmse':float(np.sqrt(mean_squared_error(test[target],p))), 'r2':r2_score(test[target],p)}
    report={'objective':objective,'analysis_type':'one-step chronological forecasting','target':target,
        'train_rows':len(train),'test_rows':len(test),'train_end':str(train.index.max()),'test_start':str(test.index.min()),
        'split':'Last 20% of targets held out; expanding-window CV and scaling confined to the earlier training period.',
        'features':features,'metrics':score(pred),'baseline':score(naive),'best_alpha':search.best_params_['model__alpha'],
        'interpretation':'Observed lags available at each forecast origin. This is rolling one-step evaluation, not an unconditional multi-day forecast or evidence of causation.'}
    summary=pd.DataFrame({'date':test.index.astype(str),'actual':test[target].to_numpy(),'prediction':pred,'last_observation_baseline':naive})
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(test.index,test[target],label='actual');ax.plot(test.index,pred,label='prediction');ax.plot(test.index,naive,label='last observation',alpha=.6);ax.legend();ax.set_title('Untouched chronological holdout');fig.tight_layout();fig.savefig(output()/'overview.png',dpi=120);plt.close(fig)
    return finish(report,summary)
