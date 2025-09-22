import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from src.logger import logging
from src.exception import CustomException


MODEL_DIR = "artifacts"
DATA_DIR = os.path.join("data", "processed")

def load_model():
    obj = joblib.load(os.path.join(MODEL_DIR,'baseline_model.joblib'))
    return obj['model']

def load_test():
    try:
        logging.info("Loading and Preparing the test data. ")
        test_path = os.path.join(DATA_DIR, "test.csv")
        df = pd.read_csv(test_path)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        logging.info("Loading and Preparation of test data has been completed successfuly.")
        return X, y
    except Exception as e:
        raise CustomException(e)
    
def precision_at_k(y_true, probs, k=0.1):
    try:
        logging.info("Calculating a business-centric metric called Precision at k.")
        n = len(probs)
        top_n = max(1, int(n * k)) #n * k: It multiplies the total number of customers (n) by the fraction k (which defaults to 0.1, or 10%). Example: If you have 1000 customers (n=1000) and k=0.1, then top_n will be 100. This means we're focusing on the 100 customers with the highest probability of churning.
        idx = np.argsort(probs)[::-1][:top_n] # np.argsort(probs): This sorts the probabilities in ascending order but returns their original indices. Result: idx now holds the indices of the 100 customers your model is most confident will churn.
        logging.info("Calculation succesfully completed.")
        return y_true.iloc[idx].sum() / top_n  
    except Exception as e:
        raise CustomException(e)  
            
def run_eval():
    try:
        logging.info("Calculating Probablities and Prediction by test data.")
        model = load_model()
        x_test, y_test = load_test()
        probs = model.predict_proba(x_test)[:,1]  # Column 0: The probability of the sample belonging to the negative class (e.g., Class 0, "No Churn"). Column 1: The probability of the sample belonging to the positive class (e.g., Class 1, "Churn"). this method returns an array with two columns:  

        preds = (probs >= 0.5).astype(int) # If your probs array is [0.1, 0.8, 0.45, 0.5], then (probs >= 0.5) would produce:[False, True, False, True]
        logging.info("Succesfully calculated probs and preds.")

        print("ROC AUC:", roc_auc_score(y_test,probs) )
        print(classification_report(y_test,preds))

        logging.info("Plotting and Saving Receiver Operating Characteristic (ROC) curve.")
        fpr,tpr,_ = roc_curve(y_test,probs)
        plt.figure(figsize=(10,10))
        plt.plot(fpr,tpr,label=f"AUC={roc_auc_score(y_test,probs):.3f}")
        plt.plot([0,1],[0,1],'--',color='grey')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("artifacts/roc_curve.png")
        plt.close()

        logging.info("Plotting and Saving precision_recall_curve.")
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig("artifacts/pr_curve.png")
        plt.close()

        logging.info("Plotting and Saving Confusion Matrix.")
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap='Blues')
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks([0,1])
        plt.yticks([0,1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()

        print("Saved ROC/PR/confusion plots to artifacts/")
        logging.info("Succesfully plotted and saved all the plots.")
    except Exception as e:
        raise CustomException(e)    

if __name__ == "__main__":
    run_eval()    
