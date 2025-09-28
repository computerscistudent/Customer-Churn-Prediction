import joblib, os, pandas as pd, numpy as np
from src.logger import logging
from src.exception import CustomException
from src.features.build_features import build_features
ART="artifacts"

class predictor:
    def __init__(self,model_path = os.path.join(ART,'best_model.joblib')):
        obj = joblib.load(model_path)
        self.model = obj['model']

        self.num_feats = obj['num_feats']
        self.cat_feats = obj['cat_feats']
        self.all_features = self.num_feats + self.cat_feats

        self.pre = self.model.named_steps['pre']
        self.clf = self.model.named_steps['clf']

    def predict_single(self, row,thresh=0.5):
            try:
                logging.info("predicting single values")
                df = pd.DataFrame([row])
                df = df.reindex(columns=self.all_features, fill_value=np.nan)
                df = build_features(df)
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
                prob = self.model.predict_proba(df)[:,1][0]
                label = int(prob>=thresh)
                logging.info("prediction successful")
                return {"probability": round(float(prob),2), "label": int(label)}
            except Exception as e:
                 raise CustomException(e)

    def explain_single(self, row: dict, top_k=5):
        try:
            logging.info("prediction values with shap library")
            import shap
            df = pd.DataFrame([row])
            df = df.reindex(columns=self.all_features, fill_value=np.nan)
            df = build_features(df)
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
            X_trans = self.pre.transform(df)
            explainer = shap.TreeExplainer(self.clf)
            shap_values = explainer.shap_values(X_trans)[0] if isinstance(explainer.shap_values(X_trans), list) else explainer.shap_values(X_trans)
            cat = self.pre.named_transformers_['cat'].named_steps['ohe']
            cat_feat_names = cat.get_feature_names_out(self.pre.transformers_[1][2])
            num_feats = self.pre.transformers_[0][2]
            feat_names = list(num_feats) + list(cat_feat_names)
            sv = pd.Series(shap_values.flatten(), index=feat_names).abs().sort_values(ascending=False)
            top = sv.head(top_k).to_dict()
            logging.info("prediction successful")
            return top        
        except Exception as e:
                 raise CustomException(e)