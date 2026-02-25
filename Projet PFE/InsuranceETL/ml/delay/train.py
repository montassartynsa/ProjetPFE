"""
ml/delay/train.py
Train delay prediction model using Random Forest and XGBoost.
Production-ready ML pipeline for insurance claim delay prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
from etl.extract import extract_table

# Configuration
MODEL_DIR = "ml/delay/models"
DATA_TABLE = "ml.ml_claim"
TARGET_COL = "is_delayed"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data():
    """Load ML dataset from database."""
    print("[LOAD] Loading ML dataset from database...")
    df = extract_table(DATA_TABLE)
    print(f"[LOAD] Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categoricals."""
    # Filter valid target rows
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Drop ID columns and non-feature columns
    drop_cols = ["claim_id", "client_id", "contract_id", "vehicle_id",
                 "date_sinistre_claim", "est_frauduleux_claim", "claim_severity_bucket"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"[PREPROCESS] Numeric features: {len(numeric_cols)}")
    print(f"[PREPROCESS] Categorical features: {len(categorical_cols)}")
    print(f"[PREPROCESS] Target distribution: {y.value_counts().to_dict()}")

    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate models."""
    models = {}
    results = {}

    # Random Forest
    print("[TRAIN] Training Random Forest...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    models['random_forest'] = rf_pipeline
    results['random_forest'] = {
        'accuracy': rf_pipeline.score(X_test, y_test),
        'auc': roc_auc_score(y_test, rf_proba),
        'report': classification_report(y_test, rf_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist()
    }

    # XGBoost
    print("[TRAIN] Training XGBoost...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
        ))
    ])

    xgb_pipeline.fit(X_train, y_train)
    xgb_pred = xgb_pipeline.predict(X_test)
    xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    models['xgboost'] = xgb_pipeline
    results['xgboost'] = {
        'accuracy': xgb_pipeline.score(X_test, y_test),
        'auc': roc_auc_score(y_test, xgb_proba),
        'report': classification_report(y_test, xgb_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist()
    }

    return models, results

def train_duration_model(df, preprocessor):
    """Train regression model for delay duration on delayed claims."""
    # Load original claims data for duration (since ml_claim drops duration)
    claims_df = extract_table("stg.clean_claims")

    # Filter delayed claims with duration data
    duration_data = claims_df[(claims_df['is_delayed'] == 1) & (claims_df['duree_traitement_jours'].notna())].copy()

    if len(duration_data) < 100:
        print("[WARNING] Insufficient duration data for training. Skipping duration model.")
        return

    print(f"[TRAIN] Training duration model on {len(duration_data)} delayed claims")

    # Prepare features same as delay model
    X_dur = duration_data.copy()
    drop_cols = ["claim_id", "client_id", "contract_id", "vehicle_id",
                 "date_sinistre_claim", "est_frauduleux_claim", "claim_severity_bucket",
                 "is_delayed", "duree_traitement_jours", "date_cloture_claim",
                 "duree_traitement_heures", "description_sinistre_claim"]
    X_dur = X_dur.drop(columns=[c for c in drop_cols if c in X_dur.columns], errors='ignore')
    y_dur = duration_data['duree_traitement_jours']

    # Train Random Forest Regressor
    dur_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        ))
    ])

    dur_pipeline.fit(X_dur, y_dur)

    # Save duration model
    joblib.dump(dur_pipeline, os.path.join(MODEL_DIR, 'duration_model.pkl'))

    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    dur_pred = dur_pipeline.predict(X_dur)
    mae = mean_absolute_error(y_dur, dur_pred)
    r2 = r2_score(y_dur, dur_pred)

    print(f"[TRAIN] Duration model - MAE: {mae:.2f} days, R²: {r2:.4f}")

def save_models(models, results):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save best model (higher AUC)
    rf_auc = results['random_forest']['auc']
    xgb_auc = results['xgboost']['auc']

    if xgb_auc > rf_auc:
        best_model = models['xgboost']
        best_name = 'xgboost'
    else:
        best_model = models['random_forest']
        best_name = 'random_forest'

    print(f"[SAVE] Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")

    # Save best model
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'delay_prediction_model.pkl'))

    # Save all models
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.pkl'))

    # Save results
    joblib.dump(results, os.path.join(MODEL_DIR, 'model_results.pkl'))

    print("[SAVE] Models and results saved to ml/delay/models/")

def main():
    """Main training pipeline."""
    print("🚀 DELAY PREDICTION MODEL TRAINING STARTED")

    # Load data
    df = load_data()

    # Preprocess
    X, y, preprocessor = preprocess_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test, preprocessor)

    # Print results
    for name, res in results.items():
        print(f"\n[{name.upper()}]")
        print(f"  Accuracy: {res['accuracy']:.4f}")
        print(f"  AUC: {res['auc']:.4f}")
        print(f"  Precision (Delayed): {res['report']['1']['precision']:.4f}")
        print(f"  Recall (Delayed): {res['report']['1']['recall']:.4f}")

    # Save models
    save_models(models, results)

    print("✅ DELAY PREDICTION MODEL TRAINING COMPLETED")

if __name__ == "__main__":
    main()