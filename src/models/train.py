import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from src.logger import logging
from src.exception import CustomException


DATA_DIR = os.path.join("data","processed")
MODEL_DIR = os.path.join("artifacts")
os.makedirs(MODEL_DIR,exist_ok=True)

def load_data():
    try:
        logging.info("Reading the train dataset.")
        tarin_path = os.path.join(DATA_DIR,"train.csv")
        if not os.path.exists(tarin_path):
            logging.info("Creating the train and test data from raw data from scratch.")
            df = pd.read_csv(os.path.join("data","raw","telco.csv"))
            logging.info("read the dataset as DataFrame")
            logging.info("train_test_split has been initialized")
            train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Churn'])
            train.to_csv(os.path.join("data","processed", "train.csv"), index=False)
            test.to_csv(os.path.join("data","processed", "test.csv"), index=False)
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        logging.info("Succesfully loaded the training dataset.") 
        return train
           
    except Exception as e:
        raise CustomException(e)
    
def prepare_features(df):
    try:
        logging.info("Preparing the data for model training.")
        if "customerID" in df.columns:
            df = df.drop(columns=['customerID'])
        if "TotalCharges" in df.columns and df["TotalCharges"].dtype == "object":
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce") 
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
        X = df.drop(columns=['Churn'])
        y = df['Churn']    
        num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist() 
        cat_feats = X.select_dtypes(include=['object','bool','category']).columns.tolist()
        logging.info("The Preparation of data is complete.")

        return X,y, num_feats,cat_feats 
    
    except Exception as e:
        raise CustomException(e)
    

def build_pipeline(num_feats, cat_feats):
    try:
        logging.info("Started building the Pipeline.")
        num_pipe = Pipeline([
            ('impute', SimpleImputer(strategy="median")),
            ('scale',StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer([
            ('num',num_pipe,num_feats),
            ('cat',cat_pipe,cat_feats)
        ])
        pipe = Pipeline([
            ('pre',preprocessor),
            ('clf',LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])
        logging.info("Pipeline has been successfuly build.")
        return pipe

    except Exception as e:
        raise CustomException(e)    
    
def run_train():
    try:
        logging.info("Implementing the Grid_Search_CV.")
        df = load_data()
        X, y, num_feats, cat_feats = prepare_features(df)
        pipe = build_pipeline(num_feats, cat_feats)
        param_grid = {
            'clf__C': [0.01, 0.1, 1.0]
        }
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X, y)
        print("Best CV score (roc_auc):", grid.best_score_)
        print("Best params:", grid.best_params_)
        logging.info("Grid_Search implementation completed succesfully. ")

        logging.info("Saving the trained Model in a file")
        model_path = os.path.join(MODEL_DIR,"baseline_model.joblib")
        joblib.dump({'model': grid.best_estimator_, 'num_feats': num_feats, 'cat_feats': cat_feats}, model_path)
        print("Saved model to", model_path)
        logging.info("Model saved successfully.")


    except Exception as e:
        raise CustomException(e)    
    
if __name__ == "__main__":
    run_train()    
    