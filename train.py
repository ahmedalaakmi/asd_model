import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib

BASE_PATH = Path("TrainingData")
ASD_PATH = BASE_PATH / "ASD"
TD_PATH = BASE_PATH / "TD"
RANDOM_STATE = 42
FEATURE_DIM = 70

def extract_features(points):
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
        *temporal, len(points)
    ]
    features = [0 if np.isnan(f) or np.isinf(f) else f for f in features]
    while len(features) < FEATURE_DIM:
        features.append(0)
    return np.array(features[:FEATURE_DIM], dtype=np.float32)

def read_scanpath_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) >= 3:
            x_vals = df.iloc[:, 1].values.astype(float)
            y_vals = df.iloc[:, 2].values.astype(float)
            if len(df.columns) >= 4:
                durations = df.iloc[:, 3].values.astype(float)
                timestamps = np.cumsum(durations) / 1000.0
            else:
                timestamps = np.arange(len(x_vals)) * 0.05
            x_vals = x_vals / 1000.0
            y_vals = y_vals / 1000.0
            x_vals = np.clip(x_vals, 0, 1)
            y_vals = np.clip(y_vals, 0, 1)
            points = np.column_stack([x_vals, y_vals, timestamps])
            if len(points) >= 10:
                return points
        return None
    except:
        return None

def main():
    print("TRAINING ASD MODEL...")
    X, y = [], []
    print("Loading ASD data...")
    for file in list(ASD_PATH.glob("*.txt")) + list(ASD_PATH.glob("*.csv")):
        points = read_scanpath_file(file)
        if points is not None:
            features = extract_features(points)
            if features is not None:
                X.append(features)
                y.append(1)
    print(f"ASD samples: {len([i for i in y if i==1])}")
    print("Loading TD data...")
    for file in list(TD_PATH.glob("*.txt")) + list(TD_PATH.glob("*.csv")):
        points = read_scanpath_file(file)
        if points is not None:
            features = extract_features(points)
            if features is not None:
                X.append(features)
                y.append(0)
    print(f"TD samples: {len([i for i in y if i==0])}")
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0:
        print("ERROR: No data loaded! Check TrainingData folder")
        return
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("Training model...")
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.1f}%")
    joblib.dump({'model': model, 'scaler': scaler, 'threshold': 0.35}, 'asd_model.pkl')
    print("Model saved to asd_model.pkl")

if __name__ == "__main__":
    main()