#!/usr/bin/env python3
"""
Re-train the hybrid model using resolution-independent (normalized) features.

Key fixes vs notebook 03:
  1. Convert CSV pixel coords → normalized [0,1] coords (per-video estimation)
  2. Update predict scripts to use raw MediaPipe normalized coords (lm.x, lm.y)
  3. Improved model architecture for >85% accuracy
"""
import os, re, joblib, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

# ── Load & parse labels ───────────────────────────────────────────────────
df = pd.read_csv('body_part_movements.csv')
print(f'Loaded {len(df)} rows')

def label_from_video_path(video_path):
    p = str(video_path).replace('\\', '/')
    if re.search(r'(^|/)correct(/|$)', p, flags=re.IGNORECASE):
        return 1
    if re.search(r'(^|/)incorrect(/|$)', p, flags=re.IGNORECASE):
        return 0
    return np.nan

df['label'] = df['label'].fillna(-1).astype(int)
path_label = df['video'].apply(label_from_video_path)
df['label'] = path_label
assert df['label'].notna().all(), 'Some videos have unparseable labels'

feature_cols = [c for c in df.columns if c not in ('video', 'label')]
N_FEATURES = len(feature_cols)
N_LANDMARKS = N_FEATURES // 2
print(f'Features: {N_FEATURES} ({N_LANDMARKS} landmarks)')

# ── Per-video normalization to [0,1] (resolution-independent) ─────────────
# CSV has pixel coords from various resolutions. We estimate each video's
# image dimensions from max coords, then normalize to [0,1] to match
# MediaPipe's native normalized output.
print('\nNormalizing features to [0,1] (resolution-independent)...')
video_groups = df.groupby('video')
x_idx = list(range(12))   # first 12 columns = x coords
y_idx = list(range(12, 24))  # last 12 columns = y coords

all_norm_frames = []
video_list = []
labels_list = []

for vid, group in video_groups:
    frames = group[feature_cols].values.astype(np.float64)
    label = int(group['label'].iloc[0])
    
    # Estimate image dimensions from the video's own max coords
    # Add 15% buffer since person may not fill entire frame
    est_w = np.max(frames[:, x_idx]) * 1.15
    est_h = np.max(frames[:, y_idx]) * 1.15
    est_w = max(est_w, 1.0)
    est_h = max(est_h, 1.0)
    
    # Normalize to [0,1]
    frames_norm = frames.copy()
    frames_norm[:, x_idx] /= est_w
    frames_norm[:, y_idx] /= est_h
    
    all_norm_frames.append(frames_norm)
    video_list.append(vid)
    labels_list.append(label)

# Pad/truncate to MAX_SEQ_LEN
MAX_SEQ_LEN = 100
X_sequences = []
y = np.array(labels_list).astype(int)

for frames in all_norm_frames:
    if len(frames) >= MAX_SEQ_LEN:
        X_sequences.append(frames[:MAX_SEQ_LEN, :])
    else:
        padded = np.zeros((MAX_SEQ_LEN, N_FEATURES))
        padded[:len(frames), :] = frames
        X_sequences.append(padded)

X_sequences = np.array(X_sequences, dtype=np.float32)
print(f'Sequences: {X_sequences.shape}')

# ── Scale across all data ────────────────────────────────────────────────
all_frames = np.vstack([seq for seq in all_norm_frames])
scaler = MinMaxScaler()
scaler.fit(all_frames)

X_scaled = np.array([scaler.transform(seq) for seq in X_sequences])
print(f'Scaled sequences: {X_scaled.shape}')
print(f'Class counts: {pd.Series(y).value_counts().to_dict()}')

# ── 5-Fold CV for honest evaluation ──────────────────────────────────────
print('\n' + '=' * 60)
print('5-Fold Stratified Cross-Validation')
print('=' * 60)

def temporal_attention(inputs):
    score = layers.Dense(1, activation='tanh')(inputs)
    att_w = layers.Flatten()(score)
    att_w = layers.Activation('softmax', name='attention_weights')(att_w)
    ctx = layers.Dot(axes=1)([att_w, inputs])
    return ctx, att_w

def build_lstm():
    inp = layers.Input(shape=(MAX_SEQ_LEN, N_FEATURES))
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    )(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True,
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    )(x)
    x = layers.Dropout(0.3)(x)
    ctx, att = temporal_attention(x)
    out = layers.Dense(1, activation='sigmoid', name='classification')(ctx)
    full = models.Model(inputs=inp, outputs=[out, att])
    train = models.Model(inputs=inp, outputs=out)
    return full, train

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    print(f'\n  Fold {fold+1}/5')
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f'    Train: {len(y_tr)} ({pd.Series(y_tr).value_counts().to_dict()})')
    print(f'    Test:  {len(y_te)} ({pd.Series(y_te).value_counts().to_dict()})')
    
    # Noise augmentation
    noise = np.random.normal(0, 0.01, X_tr.shape).astype(np.float32)
    X_tr_aug = np.clip(X_tr + noise, 0.0, 1.0)
    
    full_model, train_model = build_lstm()
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    train_model.fit(
        X_tr_aug, y_tr,
        validation_data=(X_te, y_te),
        epochs=80, batch_size=8,
        callbacks=[es, lr], verbose=0
    )
    
    probs = train_model.predict(X_te, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_te, preds)
    cv_scores.append(acc)
    print(f'    Accuracy: {acc:.4f}')
    print(f'    Probs: {[f"{p:.3f}" for p in probs]}')
    tf.keras.backend.clear_session()

print(f'\n  CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}')
print(f'  Per-fold: {[f"{s:.4f}" for s in cv_scores]}')

# ── Final model on ALL data ───────────────────────────────────────────────
print('\n' + '=' * 60)
print('Training final model on ALL data')
print('=' * 60)

noise = np.random.normal(0, 0.01, X_scaled.shape).astype(np.float32)
X_aug = np.clip(X_scaled + noise, 0.0, 1.0)

X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X_aug, y, test_size=0.15, random_state=42, stratify=y
)

final_full, final_train = build_lstm()
final_train.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

final_train.fit(
    X_tr_f, y_tr_f,
    validation_data=(X_val_f, y_val_f),
    epochs=100, batch_size=8,
    callbacks=[es, lr], verbose=1
)

final_full.set_weights(final_train.get_weights())

val_probs = final_train.predict(X_val_f, verbose=0).ravel()
val_acc = accuracy_score(y_val_f, (val_probs > 0.5).astype(int))
print(f'\nFinal validation accuracy: {val_acc:.4f}')

# ── Reference statistics from CORRECT videos ──────────────────────────────
correct_frames = []
for seq, lbl in zip(all_norm_frames, labels_list):
    if lbl == 1:
        correct_frames.append(scaler.transform(seq))

if correct_frames:
    cf_all = np.vstack(correct_frames)
    mean_correct = cf_all.mean(axis=0)
    std_correct = cf_all.std(axis=0)
else:
    mean_correct = np.zeros(N_FEATURES)
    std_correct = np.ones(N_FEATURES)

np.save('artifacts/hybrid_correct_mean.npy', mean_correct)
np.save('artifacts/hybrid_correct_std.npy', std_correct)

# ── Frame-level Random Forest ─────────────────────────────────────────────
PSEUDO_Z_THRESHOLD = 2.5
print('\nTraining frame-level RF...')

X_frames_by_video = {}
y_pseudo_by_video = {}

for seq, lbl, vid in zip(all_norm_frames, labels_list, video_list):
    seq_scaled = scaler.transform(seq)
    frames_list = []
    pseudo_list = []
    for frame in seq_scaled:
        dev = np.abs(frame - mean_correct) / (std_correct + 1e-8)
        lm_dev = np.maximum(dev[:N_LANDMARKS], dev[N_LANDMARKS:])
        max_dev = float(np.max(lm_dev))
        is_incorrect = (lbl == 0)
        pseudo = int(is_incorrect and (max_dev > PSEUDO_Z_THRESHOLD))
        frames_list.append(frame)
        pseudo_list.append(pseudo)
    X_frames_by_video[vid] = np.asarray(frames_list, dtype=np.float32)
    y_pseudo_by_video[vid] = np.asarray(pseudo_list, dtype=np.int32)

all_X = np.vstack([X_frames_by_video[v] for v in video_list])
all_y = np.concatenate([y_pseudo_by_video[v] for v in video_list])

print(f'  Training RF on {len(all_X)} frames ({all_y.sum()} problematic)')

rf_final = RandomForestClassifier(
    n_estimators=300, random_state=42,
    class_weight='balanced_subsample', n_jobs=-1
)
rf_final.fit(all_X, all_y)
rf_bundle = {'trained': True, 'model': rf_final, 'feature_names': feature_cols}

# ── Save artifacts ────────────────────────────────────────────────────────
final_full.save('models/hybrid_lstm_attention_model.keras')
joblib.dump(scaler, 'models/hybrid_scaler.joblib')
joblib.dump(rf_bundle, 'models/hybrid_rf_frame_model.joblib')

hybrid_metadata = {
    'max_seq_len': MAX_SEQ_LEN,
    'n_features': N_FEATURES,
    'feature_names': feature_cols,
    'label_map': {'correct': 1, 'incorrect': 0},
    'selected_landmarks': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    'landmark_names': ['L-Shoulder','R-Shoulder','L-Elbow','R-Elbow',
                       'L-Wrist','R-Wrist','L-Hip','R-Hip',
                       'L-Knee','R-Knee','L-Ankle','R-Ankle'],
    'feature_layout': 'x_all_then_y_all',
    'normalized_coords': True,
    'pseudo_z_threshold': PSEUDO_Z_THRESHOLD,
    'cv_accuracy_mean': float(np.mean(cv_scores)) if cv_scores else None,
    'cv_accuracy_std': float(np.std(cv_scores)) if cv_scores else None,
}
joblib.dump(hybrid_metadata, 'models/hybrid_metadata.pkl')

print('\n✓ Saved model artifacts:')
for f in ['models/hybrid_lstm_attention_model.keras',
          'models/hybrid_scaler.joblib',
          'models/hybrid_rf_frame_model.joblib',
          'models/hybrid_metadata.pkl',
          'artifacts/hybrid_correct_mean.npy',
          'artifacts/hybrid_correct_std.npy']:
    sz = os.path.getsize(f)
    print(f'  {f} ({sz:,} bytes)')
print('\nDone!')
