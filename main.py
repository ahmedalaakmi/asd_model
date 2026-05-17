from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import sqlite3
import os
import io
import time
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "asd_model.pkl"
DB_PATH = "asd_sessions.db"
FEATURE_DIM = 70
THRESHOLD = 0.35

# ─── Lifespan (load model + init DB on startup) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield

app = FastAPI(
    title="ASD Detection API",
    description="Eye-tracking based ASD screening API for Flutter app",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model state ───────────────────────────────────────────────────────
ml_model = None
ml_scaler = None

def load_model():
    global ml_model, ml_scaler
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        ml_model = data["model"]
        ml_scaler = data["scaler"]
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        # Create dummy model for testing without real training data
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import RobustScaler
        import numpy as np

        print("⚠️  asd_model.pkl not found — creating dummy model for testing")
        X_dummy = np.random.rand(100, FEATURE_DIM)
        y_dummy = np.random.randint(0, 2, 100)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X_scaled, y_dummy)
        ml_model = model
        ml_scaler = scaler
        print("⚠️  Dummy model active — replace asd_model.pkl with real trained model")

# ─── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id    TEXT,
            created_at  TEXT,
            asd_prob    REAL,
            td_prob     REAL,
            result      TEXT,
            points_count INTEGER,
            notes       TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready")

def save_session(child_id: str, asd_prob: float, td_prob: float,
                 result: str, points_count: int, notes: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO sessions (child_id, created_at, asd_prob, td_prob, result, points_count, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (child_id, datetime.utcnow().isoformat(), asd_prob, td_prob,
          result, points_count, notes))
    conn.commit()
    conn.close()

# ─── Feature Extraction ───────────────────────────────────────────────────────
def extract_features(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or len(points) < 8:
        return None

    xs = points[:, 0]
    ys = points[:, 1]
    ts = points[:, 2]

    if np.max(ts) > 0:
        ts = ts / np.max(ts)

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts) + 1e-6

    distances = np.sqrt(dx**2 + dy**2)
    velocities = distances / dt

    saccades = distances > 0.04
    n_saccades = np.sum(saccades)
    saccade_rate = n_saccades / max(len(points), 1)
    mean_saccade_amp = np.mean(distances[saccades]) if n_saccades > 0 else 0
    total_saccade_dist = np.sum(distances[saccades]) if n_saccades > 0 else 0

    fixations = distances < 0.02
    n_fixations = np.sum(fixations)
    fixation_rate = n_fixations / max(len(points), 1)
    mean_fixation_dur = np.mean(dt[fixations]) if n_fixations > 0 else 0

    mean_vel = np.mean(velocities) if len(velocities) > 0 else 0
    std_vel = np.std(velocities) if len(velocities) > 0 else 0
    max_vel = np.max(velocities) if len(velocities) > 0 else 0

    center_dist = np.sqrt((xs - 0.5)**2 + (ys - 0.5)**2)
    center_bias = np.mean(center_dist < 0.2)
    mean_center_dist = np.mean(center_dist)
    std_center_dist = np.std(center_dist)

    q_tl = np.mean((xs < 0.5) & (ys < 0.5))
    q_tr = np.mean((xs > 0.5) & (ys < 0.5))
    q_bl = np.mean((xs < 0.5) & (ys > 0.5))
    q_br = np.mean((xs > 0.5) & (ys > 0.5))

    path_length = np.sum(distances)
    scanpath_efficiency = path_length / max(np.ptp(ts), 1)

    hist, _ = np.histogramdd(np.column_stack([xs, ys]), bins=6, range=[[0,1],[0,1]])
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log(hist + 1e-6))

    temporal = [np.mean(ts), np.std(ts), np.ptp(ts), np.mean(dt), np.std(dt)]

    features = [
        np.mean(xs), np.std(xs), np.median(xs), np.ptp(xs),
        np.mean(ys), np.std(ys), np.median(ys), np.ptp(ys),
        n_saccades, saccade_rate, mean_saccade_amp, total_saccade_dist,
        n_fixations, fixation_rate, mean_fixation_dur,
        mean_vel, std_vel, max_vel,
        center_bias, mean_center_dist, std_center_dist,
        q_tl, q_tr, q_bl, q_br,
        path_length, scanpath_efficiency, entropy,
        *temporal,
        len(points)
    ]

    features = [0 if (np.isnan(f) or np.isinf(f)) else f for f in features]
    while len(features) < FEATURE_DIM:
        features.append(0)

    return np.array(features[:FEATURE_DIM], dtype=np.float32)

def parse_scanpath_csv(content: bytes) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(content))

    # Support both column formats
    if "x" in df.columns and "y" in df.columns:
        x = df["x"].values / 1000.0
        y = df["y"].values / 1000.0
        durations = df["duration"].values if "duration" in df.columns else np.ones(len(x)) * 50
    else:
        x = df.iloc[:, 1].values / 1000.0
        y = df.iloc[:, 2].values / 1000.0
        durations = df.iloc[:, 3].values if df.shape[1] >= 4 else np.ones(len(x)) * 50

    timestamps = np.cumsum(durations) / 1000.0
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    return np.column_stack([x, y, timestamps])

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    child_id: str
    asd_probability: float
    td_probability: float
    result: str          # "ASD" or "TD"
    confidence: float
    recommendation: str
    points_analyzed: int
    session_id: int

class SessionRecord(BaseModel):
    id: int
    child_id: str
    created_at: str
    asd_prob: float
    td_prob: float
    result: str
    points_count: int
    notes: Optional[str]

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": ml_model is not None,
        "endpoints": ["/predict", "/history/{child_id}", "/history", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": ml_model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Scanpath CSV file"),
    child_id: str = "unknown",
    notes: str = ""
):
    """
    Receive a scanpath CSV from Flutter app and return ASD probability.
    
    CSV format expected:
    Idx, x, y, duration
    0, 512, 300, 150
    ...
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    content = await file.read()

    try:
        points = parse_scanpath_csv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {str(e)}")

    if len(points) < 15:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough eye-tracking points: {len(points)} (need at least 15)"
        )

    # Split into chunks for more robust prediction
    n_chunks = min(3, len(points) // 10)
    chunks = np.array_split(points, max(n_chunks, 1))
    probabilities = []

    for chunk in chunks:
        if len(chunk) < 8:
            continue
        features = extract_features(chunk)
        if features is not None:
            features_scaled = ml_scaler.transform(features.reshape(1, -1))
            prob = ml_model.predict_proba(features_scaled)[0, 1]
            probabilities.append(prob)

    if not probabilities:
        raise HTTPException(status_code=422, detail="Could not extract features from data")

    asd_prob = float(np.mean(probabilities))
    td_prob = 1.0 - asd_prob
    result = "ASD" if asd_prob >= THRESHOLD else "TD"
    confidence = asd_prob if result == "ASD" else td_prob

    recommendation = (
        "Recommended follow-up with specialist"
        if result == "ASD"
        else "Routine developmental monitoring"
    )

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        INSERT INTO sessions (child_id, created_at, asd_prob, td_prob, result, points_count, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (child_id, datetime.utcnow().isoformat(), asd_prob, td_prob,
          result, len(points), notes))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return PredictResponse(
        child_id=child_id,
        asd_probability=round(asd_prob, 4),
        td_probability=round(td_prob, 4),
        result=result,
        confidence=round(confidence, 4),
        recommendation=recommendation,
        points_analyzed=len(points),
        session_id=session_id
    )


@app.get("/history/{child_id}", response_model=list[SessionRecord])
def get_child_history(child_id: str):
    """Get all sessions for a specific child."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM sessions WHERE child_id = ? ORDER BY created_at DESC",
        (child_id,)
    ).fetchall()
    conn.close()

    return [
        SessionRecord(
            id=r[0], child_id=r[1], created_at=r[2],
            asd_prob=r[3], td_prob=r[4], result=r[5],
            points_count=r[6], notes=r[7]
        ) for r in rows
    ]


@app.get("/history", response_model=list[SessionRecord])
def get_all_history(limit: int = 50):
    """Get all sessions (latest first)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()

    return [
        SessionRecord(
            id=r[0], child_id=r[1], created_at=r[2],
            asd_prob=r[3], td_prob=r[4], result=r[5],
            points_count=r[6], notes=r[7]
        ) for r in rows
    ]


@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    """Delete a specific session record."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"deleted": session_id}


# Initialize on startup
init_db()
load_model()
