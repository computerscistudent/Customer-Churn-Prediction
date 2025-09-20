import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

RAW_PATH = os.path.join("data", "raw", "telco.csv")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

def run_data_ingestion(test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(RAW_PATH)
        logging.info("read the dataset as DataFrame")
        logging.info("train_test_split has been initialized")
        train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Churn'])
        train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
        test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
        # the train and test csv files could have also been created this way as below
        # train.to_csv(os.path.join("data","processed", "train.csv"), index=False)
        # test.to_csv(os.path.join("data","processed", "test.csv"), index=False) 
        print("Wrote processed train/test to", PROC_DIR)
        logging.info("Ingestion of the data is completed")
    except Exception as e:
        raise CustomException(e)    

if __name__ == "__main__":
    run_data_ingestion()