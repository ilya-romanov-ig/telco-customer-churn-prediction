import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as mp

def main():
    train_df = pd.read_csv('../data/train_preprocessed.csv')

    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']

    scaler = StandardScaler()

    pipeline = mp(
        scaler,
        SMOTE(random_state=42),
        RandomForestClassifier(
            n_estimators=250,
            min_samples_split=12,
            min_samples_leaf=3,
            max_features='sqrt',
            max_depth=20,
            criterion='gini',
            random_state=42
        )
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, '../model/trained_model_pipeline.pkl')

if __name__ == '__main__':
    main()