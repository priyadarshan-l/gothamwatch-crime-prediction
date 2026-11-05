"""
FastAPI Backend Entrypoint for GothamWatch — Pre-Emptive Crime Prediction & Containment System
Unifies Chicago Crime Dataset Analytics, RDFLib Semantic Knowledge Graph, PyTorch ST-GAT,
and the 6-Agent Parallel Adversarial War Room.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.chicago_data_pipeline import load_or_download_chicago_data, build_spatiotemporal_windows
from backend.semantic_engine import GothamSemanticEngine
from backend.st_gnn_engine import train_and_evaluate_st_gat
from backend.gotham_agents import GothamWarRoomSwarm

app = FastAPI(
    title="GothamWatch — Crime Prediction & Autonomous Containment API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline at startup
df_chicago = load_or_download_chicago_data("data/chicago_crimes.csv")
X, Y_threat, Y_decoy, FEATURE_NAMES = build_spatiotemporal_windows(df_chicago)
st_gat_bundle = train_and_evaluate_st_gat(X, Y_threat, Y_decoy, epochs=50)
semantic_engine = GothamSemanticEngine(df_chicago)
war_room = GothamWarRoomSwarm(semantic_engine, st_gat_bundle["model"])


class SparqlRequest(BaseModel):
    query: str


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


@app.post("/api/sparql")
def execute_sparql(req: SparqlRequest):
    df_res = semantic_engine.run_sparql_to_df(req.query)
    return {"count": len(df_res), "results": df_res.to_dict(orient="records")}


@app.post("/api/scenarios/trigger")
async def trigger_scenario():
    result = await war_room.execute_full_swarm(X[-1])
    return result
