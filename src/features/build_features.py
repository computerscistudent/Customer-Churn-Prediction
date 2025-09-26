import pandas as pd
import numpy as np

def convert_total_charges(df):
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    return df

def basic_cleaning(df):
    df = df.copy()
    if "customerID" in df.columns:
            df = df.drop(columns=['customerID'])
    df = convert_total_charges(df)
    return df

def add_service_counts(df):
     services = [
        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]
     
     df["num_services"] = df[services].apply(lambda row: sum([1 if str(x).strip().lower() in ['yes','fiber optic','dsl'] and str(x).strip().lower()!="no" else 0 for x in row]), axis=1) 
     return df

def add_flags_and_feats(df):
     df['flag_totalcharges_missing'] = df['TotalCharges'].isna().astype(int)

     df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 6, 12, 24, 48, 72], labels=['0-6','7-12','13-24','25-48','49-72'])

     df['lifetime_change'] = df['tenure']*df['MonthlyCharges']

     df['contract_len'] = df['contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2}).fillna(0)

     df['pay_echeck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

     df['no_internet'] = (df['InternetService'] == 'No').astype(int)

     return df

def build_features(df):
    df = basic_cleaning(df)
    df = add_service_counts(df)
    df = add_flags_and_feats(df)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    for c in ['lifetime_charge','MonthlyCharges','TotalCharges','tenure']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df             

def get_features_list(df, target='churn'):
     
     if target in df.columns:
          df = df.drop(columns = [target])
     num_feats = df.select_dtypes(include=[np.number]).columns.tolist() 
     cat_feats = df.select_dtypes(include=['object','category','bool']).columns.tolist()     

     if 'tenure_bucket' in df.columns and 'tenure_bucket' not in cat_feats:
        cat_feats.append('tenure_bucket')
        if 'tenure_bucket' in num_feats:
            num_feats.remove('tenure_bucket')
     return sorted(num_feats), sorted(cat_feats)    