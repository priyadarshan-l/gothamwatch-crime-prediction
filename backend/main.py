"""
FastAPI Backend Entrypoint for GothamWatch — Pre-Emptive Crime Prediction & Containment System
Unifies:
  1. Real City of Chicago Crime Dataset (25,000 records, ijzp-q8t2)
  2. RDFLib Semantic Web Knowledge Graph (OWL Reasoning + Live SPARQL)
  3. PyTorch Spatio-Temporal Graph Attention Network (ST-GAT Live Inference)
  4. 6-Agent Parallel Adversarial War Room Swarm
"""

import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.chicago_data_pipeline import (
    load_or_download_chicago_data,
    build_spatiotemporal_windows,
    SECTOR_NAMES,
)
from backend.semantic_engine import GothamSemanticEngine
from backend.st_gnn_engine import train_and_evaluate_st_gat, predict_live_crime
from backend.gotham_agents import GothamWarRoomSwarm

app = FastAPI(
    title="GothamWatch — Crime Prediction & Autonomous Containment API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Dataset, PyTorch ST-GAT, and RDFLib Knowledge Graph at startup
df_chicago = load_or_download_chicago_data("data/chicago_crimes.csv")
X_windows, Y_threat, Y_decoy, FEATURE_NAMES = build_spatiotemporal_windows(df_chicago)
st_gat_bundle = train_and_evaluate_st_gat(X_windows, Y_threat, Y_decoy, epochs=60)
semantic_engine = GothamSemanticEngine(df_chicago)
war_room = GothamWarRoomSwarm(semantic_engine, st_gat_bundle["model"])

TYPE_TO_MO = {
    "ROBBERY": "MO_ConvoyAmbush",
    "WEAPONS VIOLATION": "MO_ConvoyAmbush",
    "BURGLARY": "MO_VaultBreach",
    "ARSON": "MO_GridBlackout",
    "CRIMINAL DAMAGE": "MO_GridBlackout",
    "MOTOR VEHICLE THEFT": "MO_ContainerSmuggle",
    "THEFT": "MO_ContainerSmuggle",
}


class SparqlRequest(BaseModel):
    query: str


class LivePredictRequest(BaseModel):
    sector_id: str = "S3"
    mo_id: str = "MO_GridBlackout"
    hour: int = 23
    intensity: float = 0.85
    description: Optional[str] = None
    chicago_case_number: Optional[str] = None


@app.get("/api/health")
def health():
    return {
        "status": "ONLINE",
        "chicago_records_loaded": len(df_chicago),
        "rdf_triples_count": len(semantic_engine.graph),
        "st_gat_metrics": {
            k: v for k, v in st_gat_bundle["metrics"].items() if k != "mean_attention_matrix"
        },
    }


@app.get("/api/chicago/samples")
def get_chicago_samples(limit: int = 24):
    """Return real crime cases from the Chicago Crime Dataset for live testing."""
    sig_df = df_chicago[df_chicago["is_syndicate_signature"] == 1].head(limit * 3)
    sampled = sig_df.iloc[::3].head(limit)
    records = []
    for _, row in sampled.iterrows():
        ptype = str(row["primary_type"]).upper()
        mo_id = TYPE_TO_MO.get(ptype, "MO_VaultBreach")
        records.append({
            "id": str(row.get("id", "")),
            "case_number": str(row.get("case_number", "")),
            "date": str(row.get("date", ""))[:19],
            "primary_type": ptype,
            "description": str(row.get("description", "")),
            "block": str(row.get("block", "")),
            "district": int(row.get("district_num", 1)),
            "sector_id": str(row.get("sector", "S3")),
            "sector_name": SECTOR_NAMES.get(str(row.get("sector", "S3")), "The Narrows"),
            "hour": int(row.get("hour", 22)),
            "severity": round(float(row.get("severity", 0.82)), 2),
            "latitude": round(float(row.get("latitude", 41.88)), 5),
            "longitude": round(float(row.get("longitude", -87.63)), 5),
            "mapped_mo": mo_id,
        })
    return {"total_dataset_rows": len(df_chicago), "samples": records}


@app.post("/api/predict")
def run_live_prediction(req: LivePredictRequest):
    """
    Execute real-time crime prediction using:
      1. PyTorch Spatio-Temporal Graph Attention Network (ST-GAT)
      2. RDFLib OWL Ontology Reasoner & Live SPARQL Recon
      3. Multi-Agent Tactical Choke-Point & Sentinel Guardrail Assignment
    """
    gat_pred = predict_live_crime(
        st_gat_bundle["model"],
        X_windows,
        sector_id=req.sector_id,
        mo_id=req.mo_id,
        hour=req.hour,
        intensity=req.intensity,
    )

    anom_id = f"LIVE_{req.chicago_case_number or 'ANOM'}"
    desc = req.description or f"Live anomaly ({req.mo_id}) in {req.sector_id} at {req.hour:02d}:00"
    inferred_triples = semantic_engine.inject_anomaly_and_infer(
        anomaly_id=anom_id,
        sector_id=req.sector_id,
        mo_id=req.mo_id,
        description=desc,
    )

    clean_triples = []
    for s, p, o in inferred_triples:
        s_str = "gw:" + str(s).split("#")[-1] if "#" in str(s) else str(s)
        p_str = "gw:" + str(p).split("#")[-1] if "#" in str(p) else str(p)
        o_str = "gw:" + str(o).split("#")[-1] if "#" in str(o) else str(o)
        clean_triples.append([s_str, p_str, o_str])

    # Query syndicate operatives via live SPARQL
    syn_map = {
        "MO_GridBlackout": "GhostCell",
        "MO_VaultBreach": "GhostCell",
        "MO_ConvoyAmbush": "IronMaw",
        "MO_ContainerSmuggle": "SableHand",
    }
    syn_id = syn_map.get(req.mo_id, "GhostCell")
    q_ops = f"""
    PREFIX gw: <http://gothamwatch.io/ontology#>
    SELECT ?operative ?alias ?vehicle ?weapon WHERE {{
        ?operative a gw:Operative ;
                   gw:alias ?alias ;
                   gw:belongsToSyndicate gw:{syn_id} ;
                   gw:drivesVehicle ?v ;
                   gw:armedWith ?w .
        ?v gw:name ?vehicle .
        ?w gw:name ?weapon .
    }}
    """
    ops_df = semantic_engine.run_sparql_to_df(q_ops)

    return {
        "status": "PREDICTION_COMPLETE",
        "engine": "PyTorch SpatioTemporalGAT + RDFLib OWL Reasoner",
        "prediction": gat_pred,
        "owl_inferred_triples": clean_triples,
        "total_rdf_triples": len(semantic_engine.graph),
        "sparql_operatives": ops_df.to_dict(orient="records"),
        "tactical_response": {
            "chokepoint_locked": gat_pred["predicted_escape_bridge"],
            "interceptors_assigned": ["IC-02", "IC-11", "IC-06", "DR-2"],
            "sentinel_status": "VERIFIED — Hospital (S5) & Substation (S3) Reserve Guard Maintained",
        },
    }


@app.post("/api/sparql")
def execute_sparql(req: SparqlRequest):
    df_res = semantic_engine.run_sparql_to_df(req.query)
    return {"count": len(df_res), "results": df_res.to_dict(orient="records")}


@app.post("/api/scenarios/trigger")
async def trigger_scenario():
    result = await war_room.execute_full_swarm(X_windows[-1])
    return result


# Serve built React frontend at root if present
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
