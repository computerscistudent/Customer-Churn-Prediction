import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

def convert_total_charges(df):
    try:
        logging.info("converting total charges from object to int")
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        logging.info("conversion successful")    
        return df
    except Exception as e:
        raise CustomException(e)

def basic_cleaning(df):
    try:
        logging.info("data cleaning")
        df = df.copy()
        if "customerID" in df.columns:
                df = df.drop(columns=['customerID'])
        df = convert_total_charges(df)
        logging.info("data cleaning successful")
        return df
    except Exception as e:
        raise CustomException(e)

def add_service_counts(df):
    try:
        services = [
            'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
        ]
        
        # Find which of the required service columns are actually in the input DataFrame
        existing_services = [col for col in services if col in df.columns]
        
        # If no service columns were provided at all, just set the count to 0
        if not existing_services:
            df['num_services'] = 0
            return df

        # Calculate num_services using only the columns that exist
        df['num_services'] = df[existing_services].apply(lambda row: sum([1 if str(x).strip().lower() in ['yes','fiber optic','dsl'] and str(x).strip().lower()!='no' else 0 for x in row]), axis=1)
        logging.info("service counts added succesfully")
        return df
    except Exception as e:
        raise CustomException(e)

def add_flags_and_feats(df):
     try:
        logging.info("adding the flags and features")
        df['flag_totalcharges_missing'] = df['TotalCharges'].isna().astype(int)

        df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 72], labels=['0-6','7-12','13-24','25-48','49-72'])

        df['lifetime_change'] = df['tenure']*df['MonthlyCharges']

        df['contract_len'] = df['Contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2}).fillna(0)

        df['pay_echeck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

        df['no_internet'] = (df['InternetService'] == 'No').astype(int)
        logging.info("flags and features added successfuly")

        return df
     except Exception as e:
        raise CustomException(e)

def build_features(df):
    try:
        logging.info("building features")
        df = basic_cleaning(df)
        df = add_service_counts(df)
        df = add_flags_and_feats(df)

        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        for c in ['lifetime_charge','MonthlyCharges','TotalCharges','tenure']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        logging.info("builded features successfuly")
        return df   
    except Exception as e:
        raise CustomException(e)          

def get_features_list(df, target='churn'):
     
     try:
        logging.info("creating feature list")
        if target in df.columns:
            df = df.drop(columns = [target])
        num_feats = df.select_dtypes(include=[np.number]).columns.tolist() 
        cat_feats = df.select_dtypes(include=['object','category','bool']).columns.tolist()     

        if 'tenure_bucket' in df.columns and 'tenure_bucket' not in cat_feats:
            cat_feats.append('tenure_bucket')
            if 'tenure_bucket' in num_feats:
                num_feats.remove('tenure_bucket')
        logging.info("feature list created successfuly")
        return sorted(num_feats), sorted(cat_feats)    
     except Exception as e:
        raise CustomException(e)
     




     