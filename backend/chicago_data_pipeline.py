"""
Chicago Crime Dataset Pipeline & Spatio-Temporal Feature Engineering
Maps real City of Chicago Crime records (ijzp-q8t2) into the 8-Sector GothamWatch
Spatial Graph for Spatio-Temporal Graph Attention Network (ST-GAT) training & evaluation.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

CHICAGO_API_URL = (
    "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
    "?$limit=25000&$order=date%20DESC&$where=latitude%20IS%20NOT%20NULL"
)

SECTORS = [
    {"id": "S1", "name": "Financial Core", "districts": [1, 18], "vault_value": 0.95},
    {"id": "S2", "name": "Industrial Docks", "districts": [4, 5, 22], "vault_value": 0.40},
    {"id": "S3", "name": "The Narrows", "districts": [10, 11, 15], "vault_value": 0.70},
    {"id": "S4", "name": "Diamond District", "districts": [12, 14, 19], "vault_value": 0.82},
    {"id": "S5", "name": "Old Gotham", "districts": [2, 3, 9], "vault_value": 0.60},
    {"id": "S6", "name": "Wayne Biotech", "districts": [8, 16, 17], "vault_value": 0.97},
    {"id": "S7", "name": "East End", "districts": [6, 7], "vault_value": 0.60},
    {"id": "S8", "name": "Arkham Causeway", "districts": [20, 24, 25, 31], "vault_value": 0.75},
]

SECTOR_IDS = [s["id"] for s in SECTORS]
SECTOR_NAMES = {s["id"]: s["name"] for s in SECTORS}
SECTOR_VAULT_VALUES = np.array([s["vault_value"] for s in SECTORS], dtype=np.float32)

# Map Chicago Police District numbers -> 8 Tactical Sectors
DISTRICT_TO_SECTOR = {}
for s in SECTORS:
    for d in s["districts"]:
        DISTRICT_TO_SECTOR[d] = s["id"]

ADJACENCY_LIST = {
    "S1": ["S3", "S4", "S5", "S6", "S7"],
    "S2": ["S3", "S5"],
    "S3": ["S1", "S2", "S4", "S5"],
    "S4": ["S1", "S3", "S6"],
    "S5": ["S1", "S2", "S3", "S7"],
    "S6": ["S1", "S4", "S7"],
    "S7": ["S1", "S5", "S6", "S8"],
    "S8": ["S7"],
}

SEVERITY_WEIGHTS = {
    "HOMICIDE": 1.00,
    "ROBBERY": 0.90,
    "WEAPONS VIOLATION": 0.88,
    "ARSON": 0.86,
    "BURGLARY": 0.82,
    "MOTOR VEHICLE THEFT": 0.78,
    "KIDNAPPING": 0.92,
    "ASSAULT": 0.72,
    "BATTERY": 0.70,
    "CRIMINAL SEXUAL ASSAULT": 0.90,
    "CRIMINAL DAMAGE": 0.62,
    "NARCOTICS": 0.60,
    "DECEPTIVE PRACTICE": 0.58,
    "THEFT": 0.55,
    "CRIMINAL TRESPASS": 0.48,
}


def get_adjacency_matrix() -> np.ndarray:
    """Return 8x8 symmetric adjacency matrix with self-loops."""
    n = len(SECTOR_IDS)
    adj = np.eye(n, dtype=np.float32)
    idx_map = {sid: i for i, sid in enumerate(SECTOR_IDS)}
    for u, nbrs in ADJACENCY_LIST.items():
        for v in nbrs:
            adj[idx_map[u], idx_map[v]] = 1.0
            adj[idx_map[v], idx_map[u]] = 1.0
    return adj


def load_or_download_chicago_data(csv_path: str = "data/chicago_crimes.csv") -> pd.DataFrame:
    """Load real Chicago Crime Dataset from local cache or download from City of Chicago API."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10000:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        print(f"Downloading Chicago Crime Dataset from {CHICAGO_API_URL} ...")
        req = urllib.request.Request(CHICAGO_API_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(csv_path, "wb") as out:
            out.write(resp.read())

    df = pd.read_csv(csv_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "latitude", "longitude", "primary_type"]).copy()

    # Map Chicago Police Districts to the 8 GothamWatch tactical sectors
    df["district_num"] = pd.to_numeric(df["district"], errors="coerce").fillna(1).astype(int)
    df["sector"] = df["district_num"].map(lambda d: DISTRICT_TO_SECTOR.get(d, f"S{(d % 8) + 1}"))
    df["sector_name"] = df["sector"].map(SECTOR_NAMES)

    # Extract temporal and severity attributes
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 20) | (df["hour"] <= 5)).astype(int)
    df["arrest_bool"] = df["arrest"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    df["domestic_bool"] = df["domestic"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    df["severity"] = df["primary_type"].map(lambda t: SEVERITY_WEIGHTS.get(str(t).upper(), 0.45))

    High_value_types = {"ROBBERY", "BURGLARY", "MOTOR VEHICLE THEFT", "WEAPONS VIOLATION", "ARSON"}
    df["is_syndicate_signature"] = df["primary_type"].isin(High_value_types).astype(int)

    diversion_types = {"CRIMINAL DAMAGE", "ARSON", "PUBLIC PEACE VIOLATION", "CRIMINAL TRESPASS"}
    df["is_diversion_candidate"] = df["primary_type"].isin(diversion_types).astype(int)

    return df.sort_values("date").reset_index(drop=True)


def build_spatiotemporal_windows(df: pd.DataFrame, seq_len: int = 6):
    """
    Construct spatio-temporal graph windows from real Chicago crime records.
    Returns:
      X: (N_samples, seq_len, 8, 8_features)
      Y_threat: (N_samples, 8) ground-truth next-step normalized threat per sector
      Y_decoy: (N_samples, 1) binary indicator of decoy/diversion pattern
      feature_names: list of 8 node feature names
    """
    feature_names = [
        "crime_volume_norm",
        "mean_severity",
        "syndicate_mo_ratio",
        "diversion_signal_ratio",
        "night_crime_ratio",
        "arrest_efficiency",
        "spatial_spread_std",
        "target_vault_value",
    ]

    # Bucket records into 6-hour tactical windows
    df = df.copy()
    df["time_bucket"] = df["date"].dt.floor("6h")
    buckets = sorted(df["time_bucket"].unique())

    # If records are concentrated on a few dates, also partition into chunks of ~120 incidents
    if len(buckets) < 120:
        df["bucket_id"] = df.index // 100
    else:
        bucket_map = {b: i for i, b in enumerate(buckets)}
        df["bucket_id"] = df["time_bucket"].map(bucket_map)

    num_buckets = int(df["bucket_id"].max()) + 1
    adj = get_adjacency_matrix()

    bucket_tensors = np.zeros((num_buckets, 8, 8), dtype=np.float32)
    raw_threat = np.zeros((num_buckets, 8), dtype=np.float32)

    grouped = df.groupby(["bucket_id", "sector"])
    max_count = max(1.0, float(grouped.size().max()))

    for b in range(num_buckets):
        for s_idx, sid in enumerate(SECTOR_IDS):
            if (b, sid) in grouped.groups:
                sub = grouped.get_group((b, sid))
                vol = len(sub) / max_count
                sev = float(sub["severity"].mean())
                syn_ratio = float(sub["is_syndicate_signature"].mean())
                div_ratio = float(sub["is_diversion_candidate"].mean())
                night_ratio = float(sub["is_night"].mean())
                arrest_rate = float(sub["arrest_bool"].mean())
                spread = float(np.clip(sub["latitude"].std() * 50.0, 0.0, 1.0)) if len(sub) > 1 else 0.15
            else:
                vol, sev, syn_ratio, div_ratio, night_ratio, arrest_rate, spread = (
                    0.05, 0.40, 0.10, 0.10, 0.25, 0.20, 0.10
                )

            vault_val = SECTOR_VAULT_VALUES[s_idx]
            bucket_tensors[b, s_idx] = [
                vol, sev, syn_ratio, div_ratio, night_ratio, arrest_rate, spread, vault_val
            ]
            raw_threat[b, s_idx] = np.clip(
                0.35 * vol + 0.30 * sev + 0.20 * syn_ratio + 0.15 * vault_val, 0.05, 0.98
            )

    # Create sliding windows + augment with realistic tactical decoy/contagion sequences
    rng = np.random.default_rng(42)
    X_list, Y_threat_list, Y_decoy_list = [], [], []

    for i in range(len(bucket_tensors) - seq_len - 1):
        seq_x = bucket_tensors[i : i + seq_len].copy()
        next_threat = raw_threat[i + seq_len].copy()

        # Spatial contagion across adjacent sectors
        deg = adj.sum(axis=1, keepdims=True)
        neighbor_threat = (adj @ next_threat) / deg.squeeze()
        target_threat = np.clip(0.65 * next_threat + 0.35 * neighbor_threat, 0.05, 0.98)

        # Inject decoy pattern in ~35% of windows to train the decoy head alongside normal windows
        is_decoy = 1.0 if (i % 3 == 0) else 0.0
        if is_decoy > 0.5:
            # Decoy pattern: noisy diversion spike in low/medium vault sector (e.g. S3 or S2)
            # while adjacent high-value vault sector (S6 or S1) is the true target
            decoy_sec = rng.choice([1, 2])      # S2 or S3
            true_sec = rng.choice([0, 3, 5])    # S1, S4, or S6
            seq_x[-1, decoy_sec, 0] = np.clip(seq_x[-1, decoy_sec, 0] + 0.55, 0.0, 1.0)
            seq_x[-1, decoy_sec, 3] = np.clip(seq_x[-1, decoy_sec, 3] + 0.65, 0.0, 1.0)  # High diversion signal
            seq_x[-1, true_sec, 2] = np.clip(seq_x[-1, true_sec, 2] + 0.50, 0.0, 1.0)    # High syndicate M.O.

            # True post-GAT threat suppresses decoy and elevates high-value vault
            target_threat[decoy_sec] = np.clip(target_threat[decoy_sec] * 0.55, 0.15, 0.45)
            target_threat[true_sec] = np.clip(0.82 + 0.12 * rng.random(), 0.80, 0.96)
        else:
            # Authentic crime surge in primary sector
            hot_sec = int(np.argmax(target_threat))
            seq_x[-1, hot_sec, 0] = np.clip(seq_x[-1, hot_sec, 0] + 0.45, 0.0, 1.0)
            seq_x[-1, hot_sec, 1] = np.clip(seq_x[-1, hot_sec, 1] + 0.30, 0.0, 1.0)
            target_threat[hot_sec] = np.clip(target_threat[hot_sec] + 0.25, 0.65, 0.95)

        X_list.append(seq_x)
        Y_threat_list.append(target_threat)
        Y_decoy_list.append([is_decoy])

    X = np.array(X_list, dtype=np.float32)
    Y_threat = np.array(Y_threat_list, dtype=np.float32)
    Y_decoy = np.array(Y_decoy_list, dtype=np.float32)
    return X, Y_threat, Y_decoy, feature_names
