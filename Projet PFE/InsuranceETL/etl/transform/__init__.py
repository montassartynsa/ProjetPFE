from typing import Dict, Any
import pandas as pd

from .clients import transform_clients
from .policies import transform_policies
from .vehicles import transform_vehicles
from .claims import transform_claims

from .dw_builders import (
    build_dim_client, build_dim_policy, build_dim_vehicle, build_dim_time, build_fact_claim
)
from .ml_builders import build_ml_claim_dataset


def transform_all(
    clients_raw: pd.DataFrame,
    policies_raw: pd.DataFrame,
    vehicles_raw: pd.DataFrame,
    claims_raw: pd.DataFrame
) -> Dict[str, Any]:

    clean_clients = transform_clients(clients_raw)
    clean_policies = transform_policies(policies_raw)
    clean_vehicles = transform_vehicles(vehicles_raw)
    clean_claims = transform_claims(claims_raw)

    dim_client = build_dim_client(clean_clients)
    dim_policy = build_dim_policy(clean_policies)
    dim_vehicle = build_dim_vehicle(clean_vehicles)
    dim_time = build_dim_time(clean_policies, clean_claims)

    fact_claim = build_fact_claim(clean_claims, clean_policies, clean_clients, clean_vehicles)

    ml_claim = build_ml_claim_dataset(
        clean_clients, clean_policies, clean_vehicles, clean_claims
    )

    return {
        "clean_clients": clean_clients,
        "clean_policies": clean_policies,
        "clean_vehicles": clean_vehicles,
        "clean_claims": clean_claims,

        "dim_client": dim_client,
        "dim_policy": dim_policy,
        "dim_vehicle": dim_vehicle,
        "dim_time": dim_time,
        "fact_claim": fact_claim,

        "ml_claim": ml_claim,
    }
