import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, log_loss
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings

# --- เพิ่ม Library สำหรับ LSTM ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

# ------------------------------------------------------------------------------
# 1. SETUP & UTILS
# ------------------------------------------------------------------------------
def load_csvs(folder_path, patterns):
    all_dfs = []
    for pat in patterns:
        for path in glob.glob(os.path.join(folder_path, pat)):
            try:
                df = pd.read_csv(path)
                df["source_file"] = os.path.basename(path)
                # Subsample: keep middle 60%, take every 10th frame
                df = df.sort_values("frame").iloc[int(len(df)*0.2) : int(len(df)*0.8) : 10]
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")

    if not all_dfs:
        raise ValueError("No files loaded. Check folder_path/patterns.")
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined shape: {final_df.shape}")
    return final_df

def assign_label(filename: str) -> int:
    if "_correct_" in filename or "_cor_" in filename:
        return 1
    elif "_incorrect_" in filename:
        return 0
    return np.nan

def clean_missing_coords(df, pts=33, drop_all_zero=True):
    for i in range(pts):
        if f"z{i}" not in df.columns:
            df[f"z{i}"] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan)
    required_cols = [f"x{i}" for i in range(pts)] + [f"y{i}" for i in range(pts)]
    df_clean = df.dropna(subset=required_cols)
    if drop_all_zero:
        coord_cols = required_cols + [f"z{i}" for i in range(pts)]
        all_zero_mask = (df_clean[coord_cols] == 0).all(axis=1)
        if all_zero_mask.any():
            df_clean = df_clean.loc[~all_zero_mask]
    return df_clean.reset_index(drop=True)

# ------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------------------
def preprocess_and_features(df, pts=33):
    df['label'] = df['source_file'].apply(assign_label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = clean_missing_coords(df, pts=pts)

    # Extract coords
    xs = np.vstack([df[f"x{i}"].values for i in range(pts)]).T
    ys = np.vstack([df[f"y{i}"].values for i in range(pts)]).T
    zs = np.vstack([df[f"z{i}"].values if f"z{i}" in df.columns else np.zeros(len(df)) for i in range(pts)]).T

    # Center & Scale
    center_x = xs.mean(axis=1, keepdims=True)
    center_y = ys.mean(axis=1, keepdims=True)
    center_z = zs.mean(axis=1, keepdims=True)
    
    # Torso scale
    Xc, Yc, Zc = xs - center_x, ys - center_y, zs - center_z
    torso_scale = np.linalg.norm(np.stack([Xc, Yc, Zc], axis=2), axis=2).mean(axis=1, keepdims=True)
    torso_scale = np.where(torso_scale <= 1e-6, 1.0, torso_scale)
    Xs, Ys, Zs = Xc/torso_scale, Yc/torso_scale, Zc/torso_scale

    feat_coords = np.hstack([Xs, Ys, Zs])

    # Simple Distances
    dist_to_center = np.linalg.norm(np.stack([Xs, Ys, Zs], axis=2), axis=2)
    dist_stats = np.hstack([
        dist_to_center.mean(axis=1, keepdims=True),
        dist_to_center.std(axis=1, keepdims=True),
        dist_to_center.max(axis=1, keepdims=True)
    ])

    X_feat = np.hstack([feat_coords, dist_stats])
    y = df['label'].values
    return X_feat, y, df

# ------------------------------------------------------------------------------
# 3. LSTM COMPONENT 
# ------------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq, feature)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Use last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def create_lstm_sequences(X, y, df, time_steps=5):
    """Group frames by video source and create sliding windows"""
    Xs_seq, ys_seq = [], []
    df_temp = df.copy()
    df_temp['original_index'] = range(len(df))
    
    for _, group in df_temp.groupby('source_file'):
        indices = group['original_index'].values
        if len(indices) < time_steps: continue
        for i in range(len(indices) - time_steps + 1):
            window_idxs = indices[i : i + time_steps]
            Xs_seq.append(X[window_idxs])
            ys_seq.append(y[window_idxs[-1]]) # Use label of last frame
            
    return np.array(Xs_seq), np.array(ys_seq)

def train_lstm_model(X_train, y_train, X_val, y_val, input_dim, device):
    # Hyperparams
    HIDDEN_DIM = 64
    LAYER_DIM = 2
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(input_dim, HIDDEN_DIM, LAYER_DIM, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("\n--- Training LSTM ---")
    best_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
                outputs = model(x_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    return model

# ------------------------------------------------------------------------------
# 4. MAIN PIPELINE (UPDATED)
# ------------------------------------------------------------------------------
def run_pipeline(folder_path, patterns, n_trials=10, model_choice='xgb'):
    # Load & Feats
    df = load_csvs(folder_path, patterns)
    X, y, df = preprocess_and_features(df)
    
    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- MODEL BRANCHING ---
    if model_choice == 'lstm':
        # Prepare Sequences
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TIME_STEPS = 5
        print(f"Structuring data for LSTM (Time Steps: {TIME_STEPS})...")
        X_seq, y_seq = create_lstm_sequences(Xs, y, df, time_steps=TIME_STEPS)
        
        # Split
        X_tr, X_val, y_tr, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=RANDOM_SEED)
        
        # Train
        model = train_lstm_model(X_tr, y_tr, X_val, y_val, input_dim=X_tr.shape[2], device=device)
        
        # Predict & Evaluate
        model.eval()
        with torch.no_grad():
            X_val_torch = torch.Tensor(X_val).to(device)
            probs = model(X_val_torch).cpu().numpy().flatten()
        
        y_prob = probs
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Save
        torch.save(model.state_dict(), "best_model_lstm.pth")
        joblib.dump(scaler, "scaler_lstm.joblib")
        
        print("\n" + "="*30)
        print("LSTM Evaluation Metrics")
        print("="*30)

    else:
        # Standard Machine Learning Models (Frame-based)
        classes = np.unique(y)
        class_w = compute_class_weight('balanced', classes=classes, y=y)
        sample_weight = np.array([class_w[cls] for cls in y])

        if model_choice == 'xgb':
            def obj_xgb(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
                clf = xgb.XGBClassifier(**params)
                return np.mean(cross_val_score(clf, Xs, y, scoring='roc_auc', cv=3))
            
            study = optuna.create_study(direction='maximize')
            study.optimize(obj_xgb, n_trials=n_trials)
            best_clf = xgb.XGBClassifier(**study.best_params)
            
        elif model_choice == 'lgb':
            # (Similar Optuna setup for LGBM omitted for brevity, using simple fit)
            best_clf = lgb.LGBMClassifier(n_estimators=200)

        elif model_choice == 'cat':
            best_clf = CatBoostClassifier(iterations=500, verbose=0)
            
        else:
            raise ValueError("Unknown model choice")

        # Train/Test Split
        X_tr, X_val, y_tr, y_val = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        best_clf.fit(X_tr, y_tr)
        
        # Predict
        if hasattr(best_clf, "predict_proba"):
            y_prob = best_clf.predict_proba(X_val)[:, 1]
        else:
            y_prob = best_clf.predict(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        
        joblib.dump({'model': best_clf, 'scaler': scaler}, f"best_model_{model_choice}.joblib")
        print("\n" + "="*30)
        print(f"{model_choice.upper()} Evaluation Metrics")
        print("="*30)

    # --- COMMON EVALUATION METRICS ---
    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_prob)

    print(f"Validation AUC      : {auc:.4f}")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Validation LogLoss  : {ll:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# ------------------------------------------------------------------------------
# 5. EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    path = r"../data_sample/csv_output"
    patterns = ["*_bicep_*.csv", "*any*.csv"]
    
    MODEL = 'lstm' 
    
    print(f"Starting Pipeline with {MODEL}...")
    run_pipeline(path, patterns, n_trials=5, model_choice=MODEL)