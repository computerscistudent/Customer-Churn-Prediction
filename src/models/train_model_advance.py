import os, joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from src.features.build_features import build_features, get_features_list
from src.logger import logging
from src.exception import CustomException

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)
DATA_DIR = os.path.join("data","processed")

def load_train():
    try:
        logging.info("loading training dataset")
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        df = build_features(df)
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        logging.info("dataset loaded successfuly")
        return X, y
    except Exception as e:
        raise CustomException(e)

def build_preprocessor(num_feats, cat_feats):
    try:
        logging.info("building preprocessor")
        num_pipe = Pipeline(
            [ 
                ('impute', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        ) 

        cat_pipe = Pipeline(
            [
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]
        )   

        preprocessor = ColumnTransformer(
            [
                ('num', num_pipe,num_feats),
                ('cat', cat_pipe,cat_feats)
            ]
        )

        logging.info("preprocessor builded successfuly")
        return preprocessor
    except Exception as e:
        raise CustomException(e)

def run_search_for_model(preprocessor,X,y,model_name='rf'):
    try:
        logging.info("hyperparameter testing with RF and XGB")
        if model_name == 'rf':
            estimator = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
            param_dist = {
                'clf__n_estimators': [100,200,400],
                'clf__max_depth': [None, 10, 20, 40],
                'clf__min_samples_split': [2,5,10],
                'clf__max_features': ['sqrt','log2',0.2]
            }
        else:
            estimator = XGBClassifier(use_label_encoder=False, eval_metric='auc', n_jobs=-1, random_state=42)
            param_dist = {
                'clf__n_estimators': [100,200,400],
                'clf__max_depth': [3,6,10],
                'clf__learning_rate': [0.01,0.05,0.1],
                'clf__subsample': [0.6,0.8,1.0],
                'clf__colsample_bytree': [0.6,0.8,1.0]
            }   
        pipe = Pipeline(
            [
                ('pre', preprocessor),
                ('clf', estimator)
            ]
        )     
        search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=20, cv=4,
                                    scoring='roc_auc', n_jobs=-1, verbose=2, random_state=42)
        search.fit(X,y)
        print(f"BEST for {model_name}: {search.best_score_}, params: {search.best_params_}")
        logging.info("training successful")
        return search.best_estimator_, search.best_score_
    except Exception as e:
        raise CustomException(e)

def main():
    try:
        logging.info("executing main function")
        X,y = load_train()
        num_feats, cat_feats = get_features_list(X)
        processor = build_preprocessor(num_feats,cat_feats)

        best_rf , rf_score = run_search_for_model(processor, X,y,model_name='rf')
        joblib.dump({'model': best_rf, 'num_feats': num_feats, 'cat_feats': cat_feats}, os.path.join(ARTIFACTS, 'rf_model.joblib'))

        best_xgb, xgb_score = run_search_for_model(processor, X, y, model_name='xgb')
        joblib.dump({'model': best_xgb, 'num_feats': num_feats, 'cat_feats': cat_feats}, os.path.join(ARTIFACTS, 'xgb_model.joblib'))

        if xgb_score >= rf_score:
            best, name = best_xgb, 'xgb_model.joblib'
        else:
            best, name = best_rf,  'rf_model.joblib'

        joblib.dump({'model': best, 'num_feats': num_feats, 'cat_feats': cat_feats}, os.path.join(ARTIFACTS, 'best_model.joblib'))
        print("Saved models to artifacts/. Best:", name)
        logging.info("main function executed successfuly")
    except Exception as e:
        raise CustomException(e)    


if __name__ == "__main__":
    main()    
