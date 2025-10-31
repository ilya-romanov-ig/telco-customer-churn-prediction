import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    data = pd.read_csv('../data/data_raw.csv')
    df = pd.DataFrame(data)
    df['TotalCharges'].replace(' ', None, inplace=True)
    df['TotalCharges'] = df['TotalCharges'].astype('float')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['gender'] = df['gender'].map({'Male': 1, 'Female' : 0})
    df['HasPartner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['HasDependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['HasPhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['HasPaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['HasInternetService'] = (df['InternetService'] != 'No').map({True:1, False:0})
    df['InternetType_DSL'] = ((df['InternetService'] == 'DSL') & df['HasInternetService']).map({True:1, False:0})
    df['InternetType_FiberOptic'] = ((df['InternetService'] == 'Fiber optic') & df['HasInternetService']).map({True:1, False:0})
    df['HasMultipleLines'] = ((df['MultipleLines'] == 'Yes') & df['HasPhoneService']).map({True:1, False:0})
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    
    services = [
        'OnlineSecurity', 
        'OnlineBackup', 
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    for s in services:
        df[f'Has{s}'] = ((df[s] == 'Yes') & df['HasInternetService']).map({True:1, False:0})

    one_hot_contract = pd.get_dummies(
        df['Contract'],
        prefix='ContractType'
    )

    one_hot_payment = pd.get_dummies(
        df['PaymentMethod'],
        prefix='PaymentMethodType'
    )

    df = pd.concat([df, one_hot_contract, one_hot_payment], axis=1)

    cat_cols = df.select_dtypes(include=['object']).columns
    df.drop(cat_cols, axis=1, inplace=True)
    df['AvgChargePerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Churn']
    )

    train_df.to_csv('../data/train_preprocessed.csv', index=False)
    test_df.to_csv('../data/test_preprocessed.csv', index=False)

if __name__ == '__main__':
    main()