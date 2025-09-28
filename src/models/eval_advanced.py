import os, joblib
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, average_precision_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import shap
from src.features.build_features import build_features
from sklearn.metrics import f1_score
from src.logger import logging
from src.exception import CustomException

ART = "artifacts"
DATA_DIR = os.path.join("data","processed")

def load_test():
    try:
        logging.info("loading the test data")
        df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        df = build_features(df)
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        logging.info("test data loaded succesfully")
        return X, y
    except Exception as e:
        raise CustomException(e)

def get_model(path=os.path.join(ART,'best_model.joblib')):
    try:
        logging.info("loading the model")
        obj = joblib.load(path)
        logging.info("model loaded succesfully")
        return obj['model'], obj['num_feats'], obj['cat_feats']
    except Exception as e:
        raise CustomException(e)

def best_T_for_f1(y_true, probs):
    try:
        logging.info("finding out best threshold")
        thresholds = np.linspace(0.01,0.99,99)
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs>=t).astype(int)
            f1 = f1_score(y_true,preds)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        logging.info("best threshold is found!!!")        
        return best_t, best_f1
    except Exception as e:
        raise CustomException(e)

def precission_at_k(y_true, probs, k=0.1):
    try:
        logging.info("computing precision at k")
        n = len(probs)
        top_n = max(1,int(k*n))
        idx = np.argsort(probs)[::-1][:top_n]
        logging.info("precision at k computed succesfully")
        return y_true.iloc[idx].sum() / top_n
    except Exception as e:
        raise CustomException(e)

def plot_and_save_roc_pr(y, probs, prefix='best'):
    try:
        logging.info("creating and saving plots")
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        plt.figure(figsize=(6,4))
        plt.plot(fpr,tpr,label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--',color='grey')
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(f"artifacts/{prefix}_roc.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.title("Precision-Recall")
        plt.legend()
        plt.savefig(f"artifacts/{prefix}_pr.png")
        plt.close()
        logging.info("plots created and saved successfuly")
    except Exception as e:
        raise CustomException(e)    

def shap_explain(model, X, sample_n=500, prefix='best'):
    try:
        logging.info("implementing shap and getting inside working of model")
        # model is sklearn Pipeline with 'pre' and 'clf'
        pre = model.named_steps['pre']
        clf = model.named_steps['clf']
        # transform a sample of X
        X_sample = X.sample(n=min(sample_n, len(X)), random_state=42)
        X_trans = pre.transform(X_sample)
        # get feature names
        cat = pre.named_transformers_['cat'].named_steps['ohe']
        cat_feat_names = cat.get_feature_names_out(pre.transformers_[1][2])
        num_feats = pre.transformers_[0][2]
        feat_names = list(num_feats) + list(cat_feat_names)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        # summary plot
        shap.summary_plot(shap_values, X_trans, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(f"artifacts/{prefix}_shap_summary.png")
        plt.close()
        # save top mean abs shap per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
        feat_imp.to_csv(f"artifacts/{prefix}_shap_feature_importance.csv")
        print("Saved SHAP plots and feature importance.")
        logging.info("implemented shap succesfully")
    except Exception as e:
        raise CustomException(e)    


def main():
    try:
        logging.info("executing main function")
        X_test, y_test = load_test()
        model, num_feats, cat_feats = get_model()
        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= 0.5).astype(int)

        print("ROC AUC:", roc_auc_score(y_test, probs))
        print(classification_report(y_test, preds))
        print("Precision@10%:", precission_at_k(y_test.reset_index(drop=True), probs, k=0.1))

        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        precision_for_thresholds = precision[:-1]
        idx = np.where(precision_for_thresholds >= 0.6)[0]
        chosen = thresholds[idx[-1]] if len(idx) > 0 else 0.5
        print("Chosen threshold for >=0.6 precision:", chosen)

        t, f1 = best_T_for_f1(y_test, probs)
        print("Best threshold by F1:", t, "F1:", f1)

        plot_and_save_roc_pr(y_test, probs, prefix='best')
        shap_explain(model, X_test, sample_n=500, prefix='best')
        logging.info("main function executed successfuly")
    except Exception as e:
        raise CustomException(e)    

if __name__ == "__main__":
    main()        