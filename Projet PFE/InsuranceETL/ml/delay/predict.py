"""
ml/delay/predict.py
Predict delays for claims and generate dashboard insights.
Production-ready prediction pipeline for insurance claim delay forecasting.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
from etl.extract import extract_table

# Configuration
MODEL_PATH = "ml/delay/models/delay_prediction_model.pkl"
DURATION_MODEL_PATH = "ml/delay/models/duration_model.pkl"
DATA_TABLE = "ml.ml_claim"
OUTPUT_DIR = "ml/delay"
DASHBOARD_FILE = "dashboard_data.csv"

def load_model():
    """Load trained delay prediction model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    print(f"[LOAD] Model loaded from {MODEL_PATH}")
    return model

def load_duration_model():
    """Load trained duration prediction model."""
    if os.path.exists(DURATION_MODEL_PATH):
        model = joblib.load(DURATION_MODEL_PATH)
        print(f"[LOAD] Duration model loaded from {DURATION_MODEL_PATH}")
        return model
    else:
        print("[WARNING] Duration model not found. Using average duration.")
        return None

def load_claims_data():
    """Load only ACTIVE claims for prediction."""
    print("[LOAD] Loading claims data for prediction...")
    
    # Load ML dataset
    ml_df = extract_table(DATA_TABLE)
    initial_count = len(ml_df)
    print(f"[LOAD] Loaded {initial_count} total records from {DATA_TABLE}")
    
    # Filter for active claims only
    active_statuses = ['Ouvert', 'En_cours', 'En_cours_d_expertise']
    
    if 'statut_sinistre_claim' in ml_df.columns:
        print(f"[DEBUG] Status column found. Unique values: {ml_df['statut_sinistre_claim'].unique()}")
        ml_df = ml_df[ml_df['statut_sinistre_claim'].isin(active_statuses)].copy()
        filtered_count = len(ml_df)
        closed_count = initial_count - filtered_count
        
        if filtered_count == 0:
            print(f"[WARNING] No active claims found after filtering! Using ALL claims as fallback.")
            # Reload without filtering
            ml_df = extract_table(DATA_TABLE)
            print(f"[LOAD] Using all {len(ml_df)} claims for prediction")
        else:
            print(f"[FILTER] Active: {filtered_count} | Closed (excluded): {closed_count}")
    else:
        print("[WARNING] Status column not found in ML dataset. Using ALL claims.")
    
    if len(ml_df) == 0:
        raise ValueError("No claims data available for prediction. Check database connection.")
    
    print(f"[LOAD] Processing {len(ml_df)} claims for prediction")
    return ml_df

def predict_delays(model, duration_model, df):
    """Predict delay probabilities and durations for claims."""
    if len(df) == 0:
        print("[ERROR] No claims available for prediction!")
        raise ValueError("Empty dataset - cannot make predictions")
    
    # Prepare features (same as training preprocessing)
    X = df.copy()

    # Drop non-feature columns (DO NOT drop statut_sinistre_claim - it's needed by the model)
    drop_cols = ["claim_id", "client_id", "contract_id", "vehicle_id",
                 "date_sinistre_claim", "est_frauduleux_claim", "claim_severity_bucket",
                 "is_delayed", "duree_traitement_jours"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

    # Predict probabilities
    delay_proba = model.predict_proba(X)[:, 1]  # Probability of delay (class 1)

    # Add predictions to dataframe
    df = df.copy()
    df['delay_probability'] = delay_proba

    # Classify risk levels
    df['risk_level'] = pd.cut(
        df['delay_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    # Predict duration for high-risk claims
    if duration_model is not None:
        high_risk_mask = df['delay_probability'] > 0.5
        if high_risk_mask.sum() > 0:
            X_high = X[high_risk_mask]
            predicted_durations = duration_model.predict(X_high)
            df.loc[high_risk_mask, 'predicted_delay_days'] = predicted_durations
        else:
            df['predicted_delay_days'] = 0
    else:
        # Use average duration for predicted delayed claims
        avg_duration = 30  # Default
        df['predicted_delay_days'] = np.where(df['delay_probability'] > 0.5, avg_duration, 0)

    print(f"[PREDICT] Predictions completed for {len(df)} claims")
    print(f"Risk distribution: {df['risk_level'].value_counts().to_dict()}")

    return df

def calculate_dashboard_metrics(predictions_df):
    """Calculate dashboard metrics from predictions."""
    metrics = {}

    # Risk counts
    risk_counts = predictions_df['risk_level'].value_counts()
    metrics['high_risk_count'] = int(risk_counts.get('High', 0))
    metrics['medium_risk_count'] = int(risk_counts.get('Medium', 0))
    metrics['low_risk_count'] = int(risk_counts.get('Low', 0))
    
    metrics['total_active_claims'] = len(predictions_df)

    # Estimated delayed claims (sum of probabilities)
    metrics['estimated_delayed_claims'] = float(predictions_df['delay_probability'].sum())

    # Estimated total delay days: sum of predicted delay days
    metrics['estimated_total_delay_days'] = int(predictions_df['predicted_delay_days'].sum())
    metrics['avg_predicted_delay_days'] = float(predictions_df['predicted_delay_days'].mean())

    # Cost impact: sum of claim amounts for predicted delayed claims
    # Use montant_indemnisation_claim or montant_estime_dommage_claim
    cost_cols = ['montant_indemnisation_claim', 'montant_indemnisation', 'montant_estime_dommage_claim', 'montant_estime']
    cost_col = None
    for col in cost_cols:
        if col in predictions_df.columns:
            cost_col = col
            break

    if cost_col:
        # Cost impact = sum of amounts for claims predicted to be delayed
        delayed_mask = predictions_df['delay_probability'] > 0.5
        metrics['estimated_cost_impact'] = float(predictions_df.loc[delayed_mask, cost_col].sum())
        metrics['avg_cost_per_delayed_claim'] = float(predictions_df.loc[delayed_mask, cost_col].mean())
    else:
        metrics['estimated_cost_impact'] = 0
        metrics['avg_cost_per_delayed_claim'] = 0

    # Staff recommendations
    # Assume 1 staff member can handle 50 claims per month
    # High-risk claims need priority handling
    claims_per_staff = 50
    priority_factor = 2  # High-risk claims take 2x time
    total_workload = (
        metrics['high_risk_count'] * priority_factor +
        metrics['medium_risk_count'] * 1.5 +
        metrics['low_risk_count']
    )
    metrics['recommended_staff'] = max(1, int(np.ceil(total_workload / claims_per_staff)))

    # Time-based projections (next month)
    current_month = datetime.now().month
    next_month = (datetime.now() + timedelta(days=30)).month
    metrics['projection_month'] = next_month

    print("[METRICS] Dashboard metrics calculated:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics

def generate_dashboard_data(predictions_df, metrics):
    """Generate comprehensive dashboard data."""
    # Top high-risk claims
    high_risk_claims = predictions_df[predictions_df['risk_level'] == 'High'].copy()
    high_risk_claims = high_risk_claims.nlargest(10, 'delay_probability')

    # Select available columns for high-risk claims
    desired_cols = ['claim_id', 'delay_probability', 'montant_estime', 'montant_indemnisation',
                    'type_sinistre', 'client_id', 'contract_id', 'predicted_delay_days']
    available_cols = [c for c in desired_cols if c in high_risk_claims.columns]
    high_risk_data = high_risk_claims[available_cols].to_dict('records') if not high_risk_claims.empty else []

    # Claims by risk level with key info
    dashboard_data = {
        'summary': metrics,
        'high_risk_claims': high_risk_data,
        'risk_distribution': predictions_df['risk_level'].value_counts().to_dict(),
        'monthly_projection': {
            'month': metrics['projection_month'],
            'estimated_delays': metrics['estimated_delayed_claims'],
            'cost_impact': metrics['estimated_cost_impact'],
            'staff_needed': metrics['recommended_staff']
        }
    }

    return dashboard_data

def save_dashboard_data(dashboard_data):
    """Save dashboard data to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as JSON for dashboard consumption
    import json
    output_path = os.path.join(OUTPUT_DIR, 'dashboard_insights.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    # Also save predictions CSV
    predictions_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    dashboard_data['predictions'] = dashboard_data.pop('summary', {})  # For CSV
    # Actually, save the full predictions separately if needed

    print(f"[SAVE] Dashboard data saved to {output_path}")

def main():
    """Main prediction pipeline."""
    print("[START] DELAY PREDICTION & DASHBOARD GENERATION STARTED")

    try:
        # Load model and data
        model = load_model()
        duration_model = load_duration_model()
        claims_df = load_claims_data()

        # Make predictions
        predictions_df = predict_delays(model, duration_model, claims_df)

        # Calculate metrics
        metrics = calculate_dashboard_metrics(predictions_df)

        # Generate dashboard data
        dashboard_data = generate_dashboard_data(predictions_df, metrics)

        # Save results
        save_dashboard_data(dashboard_data)

        print("[SUCCESS] DELAY PREDICTION & DASHBOARD GENERATION COMPLETED")

        # Print key insights
        print("\n[INSIGHTS] KEY DASHBOARD INSIGHTS:")
        print(f"High-risk claims: {metrics['high_risk_count']}")
        print(f"Estimated delayed claims: {metrics['estimated_delayed_claims']:.1f}")
        print(f"Estimated total delay days: {metrics['estimated_total_delay_days']:.0f}")
        print(f"Estimated cost impact: ${metrics['estimated_cost_impact']:,.2f}")
        print(f"Recommended staff: {metrics['recommended_staff']}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise

if __name__ == "__main__":
    main()
