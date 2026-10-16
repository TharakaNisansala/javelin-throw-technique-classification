#!/usr/bin/env python3
"""
Flask web app for video-based javelin throw analysis.
Users upload a video, the system classifies it as CORRECT or INCORRECT,
and shows every frame with annotated body landmarks:
  - Green  = body part is within normal range
  - Red    = body part deviates from correct pattern (z-score > threshold)
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_lstm_attention_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "hybrid_scaler.joblib")
RF_PATH = os.path.join(BASE_DIR, "models", "hybrid_rf_frame_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "hybrid_metadata.pkl")
MEAN_PATH = os.path.join(BASE_DIR, "artifacts", "hybrid_correct_mean.npy")
STD_PATH = os.path.join(BASE_DIR, "artifacts", "hybrid_correct_std.npy")
LANDMARKER_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")

# ─── Technique RF Model paths ───────────────────────────────────────────────
RF_TECHNIQUE_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
RF_TECHNIQUE_SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
RF_TECHNIQUE_LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
RF_TECHNIQUE_CLASS_NAMES = ["BothErrors", "GoodTechnique", "LowArm", "PoorLeftLegBlock"]
RF_TECHNIQUE_COLORS = {
    "BothErrors":       "#ef4444",
    "GoodTechnique":    "#22c55e",
    "LowArm":           "#f59e0b",
    "PoorLeftLegBlock": "#a855f7",
}
# 13 landmarks used by the technique RF model (52 features = 13 × 4 coords)
RF_TECHNIQUE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
RF_TECHNIQUE_N_FEATURES = len(RF_TECHNIQUE_LANDMARK_INDICES) * 4  # 52
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv"}

# MediaPipe Pose landmark indices for the 12 body parts used by training.
LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
N_LANDMARKS = len(LANDMARK_INDICES)  # 12
N_FEATURES = N_LANDMARKS * 2        # 24

LANDMARK_NAMES = [
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
]

# Skeleton connections between the 12 tracked landmarks (by index into LANDMARK_INDICES).
# Each tuple is (left_idx, right_idx) mapping to the actual MediaPipe landmark numbers.
SKELETON_CONNECTIONS = [
    (11, 12),   # shoulders
    (11, 13),   # L-shoulder → L-elbow
    (12, 14),   # R-shoulder → R-elbow
    (13, 15),   # L-elbow → L-wrist
    (14, 16),   # R-elbow → R-wrist
    (11, 23),   # L-shoulder → L-hip
    (12, 24),   # R-shoulder → R-hip
    (23, 24),   # hips
    (23, 25),   # L-hip → L-knee
    (24, 26),   # R-hip → R-knee
    (25, 27),   # L-knee → L-ankle
    (26, 28),   # R-knee → R-ankle
]
# Map (mp_landmark_a, mp_landmark_b) → (our_index_a, our_index_b)
# our_index = position in LANDMARK_INDICES list
MP_TO_OUR = {mp_idx: i for i, mp_idx in enumerate(LANDMARK_INDICES)}
OUR_SKELETON = []
for mp_a, mp_b in SKELETON_CONNECTIONS:
    if mp_a in MP_TO_OUR and mp_b in MP_TO_OUR:
        OUR_SKELETON.append((MP_TO_OUR[mp_a], MP_TO_OUR[mp_b]))

MAX_SEQ_LEN = 100

# ─── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
app.config["UPLOAD_FOLDER"] = os.path.join(SESSIONS_DIR, "uploads")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ─── Global model references (lazy-loaded) ─────────────────────────────────────
_model = None
_scaler = None
_rf_model = None
_mean_correct = None
_std_correct = None
_landmarker = None
_rf_technique_model = None
_rf_technique_scaler = None
_rf_technique_label_encoder = None


def _load_models():
    """Lazy-load all hybrid model artifacts once."""
    global _model, _scaler, _rf_model, _mean_correct, _std_correct
    if _model is not None:
        return
    required = [
        (MODEL_PATH, "LSTM model"),
        (SCALER_PATH, "scaler"),
        (MEAN_PATH, "reference mean"),
        (STD_PATH, "reference std"),
        (LANDMARKER_PATH, "pose landmarker"),
    ]
    missing = [name for path, name in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

    _model = tf.keras.models.load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _mean_correct = np.load(MEAN_PATH)
    _std_correct = np.load(STD_PATH)

    rf_bundle = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else {}
    _rf_model = rf_bundle.get("model") if isinstance(rf_bundle, dict) else None


def _init_landmarker():
    global _landmarker
    if _landmarker is not None:
        return
    _BaseOptions = mp.tasks.BaseOptions
    _PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    _RunningMode = mp.tasks.vision.RunningMode
    options = _PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=_RunningMode.IMAGE,
    )
    _landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _load_technique_rf():
    """Lazy-load technique RF classifier, scaler, and label encoder."""
    global _rf_technique_model, _rf_technique_scaler, _rf_technique_label_encoder
    if _rf_technique_model is not None:
        return
    required = [
        (RF_TECHNIQUE_PATH, "technique RF model"),
        (RF_TECHNIQUE_SCALER_PATH, "technique scaler"),
        (RF_TECHNIQUE_LABEL_ENCODER_PATH, "technique label encoder"),
    ]
    missing = [name for path, name in required if not os.path.exists(path)]
    if missing:
        print(f"Warning: technique classifier not available (missing: {', '.join(missing)})")
        return

    _rf_technique_model = joblib.load(RF_TECHNIQUE_PATH)
    _rf_technique_scaler = joblib.load(RF_TECHNIQUE_SCALER_PATH)
    _rf_technique_label_encoder = joblib.load(RF_TECHNIQUE_LABEL_ENCODER_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Core processing functions ──────────────────────────────────────────────────

def extract_landmarks(frame):
    """
    Run MediaPipe pose detection on a frame.
    Returns:
        features: 24-element array [x0...x11, y0...y11] or None
        landmarks_px: list of (x_pixel, y_pixel) for the 12 tracked landmarks, or None
        technique_features: 52-element array [x0,y0,z0,vis0, ...] for 13 technique landmarks, or None
    """
    global _landmarker
    if _landmarker is None:
        return None, None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None, None, None

    h, w = frame.shape[:2]
    lms = result.pose_landmarks[0]
    xs, ys, landmarks_px = [], [], []
    for idx in LANDMARK_INDICES:
        lm = lms[idx]
        xs.append(lm.x)
        ys.append(lm.y)
        landmarks_px.append((int(lm.x * w), int(lm.y * h)))

    features = np.array(xs + ys, dtype=np.float32)

    # Technique RF features: 13 landmarks × 4 coords (x, y, z, visibility)
    tf_vec = np.empty(RF_TECHNIQUE_N_FEATURES, dtype=np.float32)
    for i, idx in enumerate(RF_TECHNIQUE_LANDMARK_INDICES):
        lm = lms[idx]
        base = i * 4
        tf_vec[base]     = lm.x
        tf_vec[base + 1] = lm.y
        tf_vec[base + 2] = lm.z
        tf_vec[base + 3] = lm.visibility

    return features, landmarks_px, tf_vec


def compute_landmark_deviation(seq_scaled, mean_correct, std_correct):
    """Return array of shape (n_frames, n_landmarks) — max z-score of x/y per landmark."""
    z = np.abs((seq_scaled - mean_correct) / (std_correct + 1e-8))
    z_x = z[:, :N_LANDMARKS]
    z_y = z[:, N_LANDMARKS:]
    return np.maximum(z_x, z_y)


def process_video(video_path, session_dir, prob_threshold=0.40, z_threshold=2.0, skip_frames=0):
    """
    Full processing pipeline for a single video.
    Returns a dict with results and generates annotated frame images on disk.
    """
    _load_models()
    _init_landmarker()

    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # ── Step A: Extract features and landmark positions per frame ──────────
    cap = cv2.VideoCapture(video_path)
    raw_features = []
    raw_technique_features = []  # 52-dim vectors for technique RF classifier
    all_landmarks_px = []  # list of list of (x, y) tuples
    original_frames = []   # store (frame_idx, bgr_frame) for annotation

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        features, landmarks_px, technique_features = extract_landmarks(frame)
        if features is not None and landmarks_px is not None:
            raw_features.append(features)
            raw_technique_features.append(technique_features)
            all_landmarks_px.append(landmarks_px)
            original_frames.append(frame)
    cap.release()

    if len(raw_features) == 0:
        return {"error": "No valid frames processed. Check that the video contains visible people."}

    seq_all = np.array(raw_features, dtype=np.float32)
    total_frames = len(seq_all)

    # ── Skip initial frames ────────────────────────────────────────────────
    skip = min(skip_frames, total_frames - 1)
    seq_work = seq_all[skip:]
    tf_all = np.array(raw_technique_features, dtype=np.float32)
    tf_work = tf_all[skip:]
    lm_work = all_landmarks_px[skip:]
    frames_work = original_frames[skip:]
    n_work = len(seq_work)

    # ── Step B: LSTM prediction ────────────────────────────────────────────
    if n_work >= MAX_SEQ_LEN:
        chunk = seq_work[:MAX_SEQ_LEN]
        n_valid = MAX_SEQ_LEN
    else:
        chunk = np.zeros((MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
        chunk[:n_work] = seq_work
        n_valid = n_work

    chunk_scaled = _scaler.transform(chunk)
    result = _model.predict(chunk_scaled[np.newaxis, ...], verbose=0)

    if isinstance(result, list) and len(result) >= 1:
        prob = float(result[0][0, 0])
    else:
        prob = float(result[0, 0])

    # Extract attention weights
    attention_map = np.zeros(n_work, dtype=np.float32)
    if isinstance(result, list) and len(result) >= 2:
        att = result[1][0]
        attention_map[:min(n_work, MAX_SEQ_LEN)] = att[:min(n_work, MAX_SEQ_LEN)]

    overall_label = "correct" if prob >= prob_threshold else "incorrect"

    # ── Step C: Per-frame deviation analysis ───────────────────────────────
    seq_scaled = _scaler.transform(seq_work)
    lm_dev = compute_landmark_deviation(seq_scaled, _mean_correct, _std_correct)

    # RF per-frame anomaly probability
    rf_prob = np.zeros(n_work, dtype=np.float32)
    if _rf_model is not None:
        rf_prob = _rf_model.predict_proba(seq_scaled)[:, 1]

    # Normalize attention to [0,1]
    att_min, att_max = attention_map.min(), attention_map.max()
    att_norm = (attention_map - att_min) / (att_max - att_min + 1e-8)

    # Normalize deviation to [0,1]
    dev_max = lm_dev.max(axis=1)
    dev_min_v, dev_max_v = dev_max.min(), dev_max.max()
    dev_norm = (dev_max - dev_min_v) / (dev_max_v - dev_min_v + 1e-8)

    # Combined score
    combined = 0.55 * dev_norm + 0.30 * rf_prob + 0.15 * att_norm

    # ── Technique RF classification (52-dim features → 4 technique classes) ──
    _load_technique_rf()
    technique_results = None
    if _rf_technique_model is not None and _rf_technique_scaler is not None:
        # Impute any NaN in technique features
        tf_clean = tf_work.copy()
        nan_mask = np.isnan(tf_clean)
        if nan_mask.any():
            tf_clean[nan_mask] = 0.0  # safe fallback for rare NaN

        # Scale and predict
        tf_scaled = _rf_technique_scaler.transform(tf_clean)
        tf_preds = _rf_technique_model.predict(tf_scaled)
        tf_probs = _rf_technique_model.predict_proba(tf_scaled)

        # Decode integer labels
        tf_labels = _rf_technique_label_encoder.inverse_transform(tf_preds)

        # Majority vote (overall class)
        vote_counter = Counter(tf_labels)
        overall_technique = vote_counter.most_common(1)[0][0]
        overall_technique_count = vote_counter.most_common(1)[0][1]
        overall_technique_pct = overall_technique_count / n_work * 100

        # Per-class average probabilities
        avg_technique_probs = tf_probs.mean(axis=0).tolist()

        technique_results = {
            "overall_class": overall_technique,
            "overall_count": int(overall_technique_count),
            "overall_pct": round(overall_technique_pct, 1),
            "avg_probs": {cls: round(float(avg_technique_probs[i]), 4)
                          for i, cls in enumerate(RF_TECHNIQUE_CLASS_NAMES)},
            "class_names": RF_TECHNIQUE_CLASS_NAMES,
        }

    # ── Step D: Determine frame thresholds and problematic landmarks ────────
    if overall_label == "incorrect":
        frame_threshold = float(np.percentile(combined, 90))
        flagged_idx = np.where(combined >= frame_threshold)[0]
        sorted_order = np.argsort(combined[flagged_idx])[::-1]
    else:
        frame_threshold = 0.0
        flagged_idx = np.array([], dtype=int)
        sorted_order = np.array([], dtype=int)

    # ── Step E: Generate annotated frames ──────────────────────────────────
    frame_results = []
    for f_idx in range(n_work):
        orig_frame = frames_work[f_idx]
        landmarks_px = lm_work[f_idx]
        h, w = orig_frame.shape[:2]

        # Determine which landmarks are "bad" for this frame
        bad_mask = lm_dev[f_idx] > z_threshold
        bad_indices = set(np.where(bad_mask)[0])

        # Draw on a copy of the frame
        annotated = orig_frame.copy()

        # Draw skeleton connections first (so they're behind the dots)
        for our_a, our_b in OUR_SKELETON:
            if our_a < len(landmarks_px) and our_b < len(landmarks_px):
                pt1 = landmarks_px[our_a]
                pt2 = landmarks_px[our_b]
                # Determine connection color: red if either endpoint is bad
                is_bad = our_a in bad_indices or our_b in bad_indices
                color = (0, 0, 255) if is_bad else (0, 255, 0)
                cv2.line(annotated, pt1, pt2, color, 2)

        # Draw landmark dots
        for lm_i, (lx, ly) in enumerate(landmarks_px):
            is_bad = lm_i in bad_indices
            color = (0, 0, 255) if is_bad else (0, 255, 0)
            cv2.circle(annotated, (lx, ly), 6, color, -1)
            # Add a subtle outer ring
            cv2.circle(annotated, (lx, ly), 8, color, 1)

        # Write frame info
        orig_idx = f_idx + skip
        is_flagged = f_idx in set(flagged_idx)
        frame_info = {
            "frame_index": f_idx,
            "original_frame": orig_idx,
            "is_flagged": is_flagged,
            "combined_score": float(combined[f_idx]),
            "deviation": float(dev_max[f_idx]),
            "rf_prob": float(rf_prob[f_idx]),
            "bad_landmarks": [int(i) for i in bad_indices],
            "bad_landmark_names": [LANDMARK_NAMES[int(i)] for i in bad_indices],
        }

        # Save annotated frame as JPEG
        frame_filename = f"frame_{f_idx:06d}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_results.append(frame_info)

    # Determine which frames to highlight
    if overall_label == "incorrect" and len(flagged_idx) > 0:
        flagged_frames = [int(f) for f in flagged_idx[sorted_order]]
    else:
        flagged_frames = []

    results = {
        "video_name": os.path.basename(video_path),
        "total_frames": total_frames,
        "analysis_frames": n_work,
        "skip_frames": skip,
        "probability": round(prob, 4),
        "threshold": prob_threshold,
        "label": overall_label,
        "z_threshold": z_threshold,
        "frame_threshold": round(frame_threshold, 4),
        "flagged_frame_indices": flagged_frames[:100],  # limit to top 100
        "frames": frame_results,
        "landmark_names": LANDMARK_NAMES,
        "technique": technique_results,  # technique RF classification
    }

    return results


def save_results(session_dir, results):
    """Save processing results as JSON."""
    # Convert numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(session_dir, "results.json"), "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)


def load_results(session_dir):
    """Load processing results from JSON."""
    path = os.path.join(session_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def cleanup_old_sessions(max_age_hours=24):
    """Remove session directories older than max_age_hours."""
    import time
    now = time.time()
    for sess_id in os.listdir(SESSIONS_DIR):
        sess_path = os.path.join(SESSIONS_DIR, sess_id)
        if not os.path.isdir(sess_path):
            continue
        age_hours = (now - os.path.getmtime(sess_path)) / 3600
        if age_hours > max_age_hours:
            import shutil
            shutil.rmtree(sess_path, ignore_errors=True)


# ─── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video upload, process it, and redirect to results page."""
    if "video" not in request.files:
        return render_template("index.html", error="No video file selected.")

    file = request.files["video"]
    if file.filename == "":
        return render_template("index.html", error="No video file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", error=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}")

    # Parse optional parameters from form
    threshold = float(request.form.get("threshold", 0.40))
    z_threshold = float(request.form.get("z_threshold", 2.0))
    skip_frames = int(request.form.get("skip_frames", 0))
    top_frames = int(request.form.get("top_frames", 15))

    # Create a unique session
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save uploaded video
    filename = secure_filename(file.filename)
    video_path = os.path.join(session_dir, filename)
    file.save(video_path)

    try:
        # Process the video
        results = process_video(
            video_path, session_dir,
            prob_threshold=threshold,
            z_threshold=z_threshold,
            skip_frames=skip_frames,
        )
        if "error" in results:
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)
            return render_template("index.html", error=results["error"])

        # Save results
        save_results(session_dir, results)

        # Cleanup old sessions periodically
        cleanup_old_sessions()

        return redirect(url_for("results", session_id=session_id))

    except Exception as e:
        import shutil
        import traceback
        traceback.print_exc()
        shutil.rmtree(session_dir, ignore_errors=True)
        return render_template("index.html", error=f"Processing error: {str(e)}")


@app.route("/results/<session_id>")
def results(session_id):
    """Display results page for a given session."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_dir):
        return render_template("index.html", error="Session not found. Please upload a video again.")

    results_data = load_results(session_dir)
    if results_data is None:
        return render_template("index.html", error="Results not found for this session.")

    # Pre-compute most-frequently-problematic landmarks for sidebar
    landmark_counts = {}
    flagged_indices = results_data.get("flagged_frame_indices", [])[:30]
    frames = results_data.get("frames", [])
    for f_idx in flagged_indices:
        if f_idx < len(frames):
            for lm_name in frames[f_idx].get("bad_landmark_names", []):
                landmark_counts[lm_name] = landmark_counts.get(lm_name, 0) + 1
    top_landmarks = sorted(landmark_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    flagged_display = results_data.get("flagged_frame_indices", [])[:30]

    return render_template(
        "results.html",
        session_id=session_id,
        results=results_data,
        top_landmarks=top_landmarks,
        flagged_display=flagged_display,
    )


@app.route("/frames/<session_id>/<frame_idx>")
def serve_frame(session_id, frame_idx):
    """Serve an annotated frame image."""
    frame_path = os.path.join(SESSIONS_DIR, session_id, "frames", f"frame_{int(frame_idx):06d}.jpg")
    if not os.path.exists(frame_path):
        return "", 404
    return send_file(frame_path, mimetype="image/jpeg")


@app.route("/api/results/<session_id>")
def api_results(session_id):
    """Return results as JSON for API consumers."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_dir):
        return jsonify({"error": "Session not found"}), 404
    results_data = load_results(session_dir)
    if results_data is None:
        return jsonify({"error": "Results not found"}), 404
    return jsonify(results_data)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Javelin Throw Analysis Web App")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    print(f"  Javelin Throw Analysis Web App")
    print(f"  ─────────────────────────────")
    print(f"  Models loaded on first upload")
    print(f"  Open http://{args.host}:{args.port} in your browser")
    print()
    app.run(host=args.host, port=args.port, debug=args.debug)
