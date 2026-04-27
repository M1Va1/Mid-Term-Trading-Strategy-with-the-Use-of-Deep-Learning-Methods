"""
(LSTM, GRU, Transformer, LSTM+XGB, RF, XGBoost), train/val/test split,
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return F.mse_loss(pred, target)


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    cov = (pred_c * target_c).mean()
    std_pred = pred_c.std() + 1e-8
    std_target = target_c.std() + 1e-8
    corr = cov / (std_pred * std_target)
    return -corr


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Pairwise ranking loss: pred_i > pred_j when target_i > target_j."""
    n = len(pred)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    if n > 32:
        idx = torch.randperm(n)[:32]
        pred = pred[idx]
        target = target[idx]
    
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = (target.unsqueeze(1) - target.unsqueeze(0)).sign()
    loss = F.relu(-diff_pred * diff_target + margin)
    return loss.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                  mse_weight: float = 0.5, ic_weight: float = 0.5) -> torch.Tensor:
    """Combined MSE + IC loss."""
    return mse_weight * mse_loss(pred, target) + ic_weight * ic_loss(pred, target)


LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "ic": ic_loss,
    "ranking": ranking_loss,
    "combined": combined_loss,
}


def cross_sectional_normalize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:

    df = df.copy()
    df[feature_cols] = df.groupby("date")[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df


def normalize_target_monthly(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df.groupby("date")[target_col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    return df


class ReturnModel(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feature_cols: List[str]) -> "ReturnModel":
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        pass


def train_val_test_split(
    df: pd.DataFrame,
    val_start: str = "2018-01-01",
    test_start: str = "2019-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < val_start].copy()
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def compute_ic(df: pd.DataFrame, pred_col: str = "pred") -> float:
    df = df.dropna(subset=[pred_col, "target"]).copy()
    df["month"] = df["date"].dt.to_period("M")
    ics = []
    for _, group in df.groupby("month"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group[pred_col], group["target"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def compute_icir(df: pd.DataFrame, pred_col: str = "pred") -> float:
    df = df.dropna(subset=[pred_col, "target"]).copy()
    df["month"] = df["date"].dt.to_period("M")
    ics = []
    for _, group in df.groupby("month"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group[pred_col], group["target"])
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 2:
        return 0.0
    mean_ic = np.mean(ics)
    std_ic = np.std(ics) + 1e-8
    return float(mean_ic / std_ic)


def compute_metric(df: pd.DataFrame, pred_col: str = "pred", 
                   metric: str = "ic") -> float:
    if metric == "icir":
        return compute_icir(df, pred_col)
    return compute_ic(df, pred_col)


class MonthlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], lookback: int = 6):
        self.lookback = lookback
        self.feature_cols = feature_cols
        n_features = len(feature_cols)

        df = df.sort_values(["asset", "date"]).reset_index(drop=True)

        sequences, targets, indices = [], [], []
        for asset, grp in df.groupby("asset"):
            if len(grp) < lookback:
                continue
            vals = grp[feature_cols].values.astype(np.float32)
            tgts = grp["target"].values.astype(np.float32)
            idxs = grp.index.values
            for i in range(lookback, len(grp) + 1):
                seq = vals[i - lookback:i]
                sequences.append(seq)
                targets.append(tgts[i - 1])
                indices.append(idxs[i - 1])

        if sequences:
            self.X = torch.from_numpy(np.stack(sequences))
            self.y = torch.from_numpy(np.array(targets))
            self.indices = np.array(indices)
        else:
            self.X = torch.empty(0, lookback, n_features)
            self.y = torch.empty(0)
            self.indices = np.array([], dtype=int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ReturnLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 16),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class LSTMReturnModel(ReturnModel):
    def __init__(self, lookback=6, hidden_size=32, num_layers=1,
                 dropout=0.3, epochs=60, patience=10,
                 batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_df, val_df, feature_cols):
        n_features = len(feature_cols)
        train_ds = MonthlyDataset(train_df, feature_cols, self.lookback)
        val_ds = MonthlyDataset(val_df, feature_cols, self.lookback)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("LSTM: not enough sequential data, skipping")
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.model = ReturnLSTM(n_features, self.hidden_size,
                                self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        best_vl, best_w, no_imp = float("inf"), None, 0
        self.train_losses_ = []
        self.val_losses_ = []

        for _ in range(self.epochs):
            self.model.train()
            epoch_tl = []
            for X, y in train_dl:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_func(self.model(X), y)
                loss.backward()
                opt.step()
                epoch_tl.append(loss.item())
            self.train_losses_.append(np.mean(epoch_tl))

            self.model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.model(X.to(self.device)),
                                   y.to(self.device)).item() for X, y in val_dl])
            self.val_losses_.append(vl)
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        return self

    def predict(self, df, feature_cols, history_df=None):
        if self.model is None:
            return np.full(len(df), np.nan)

        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds = MonthlyDataset(combined, feature_cols, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        dl = DataLoader(ds, batch_size=512)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in dl:
                preds.append(self.model(X.to(self.device)).cpu().numpy())
        pred_arr = np.concatenate(preds)
        result_full = np.full(len(combined), np.nan)
        result_full[ds.indices] = pred_arr
        out = result_full[n_history:]
        nan_mask = np.isnan(out)
        if nan_mask.any() and (~nan_mask).any():
            out[nan_mask] = np.nanmedian(out)
        return out


class HybridDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], lookback: int = 12):
        self.lookback = lookback
        self.feature_cols = feature_cols

        df = df.sort_values(["asset", "date"]).reset_index(drop=True)

        ret_sequences = []
        flat_features = []
        targets = []
        indices = []

        for asset, grp in df.groupby("asset"):
            if len(grp) < lookback + 1:
                continue

            rets = grp["target"].values.astype(np.float32)
            feats = grp[feature_cols].values.astype(np.float32)
            idxs = grp.index.values

            for i in range(lookback, len(grp)):
                # rolling z-score over past lookback only — no look-ahead
                past_rets = rets[i - lookback:i]
                ret_mean = past_rets.mean()
                ret_std = past_rets.std()
                if ret_std < 1e-8:
                    ret_std = 1.0
                past_z_rets = (past_rets - ret_mean) / ret_std

                ret_sequences.append(past_z_rets.reshape(-1, 1))
                flat_features.append(feats[i])
                targets.append(rets[i])
                indices.append(idxs[i])

        if ret_sequences:
            self.X_seq = torch.from_numpy(np.stack(ret_sequences))
            self.X_flat = torch.from_numpy(np.stack(flat_features))
            self.y = torch.from_numpy(np.array(targets))
            self.indices = np.array(indices)
        else:
            self.X_seq = torch.empty(0, lookback, 1)
            self.X_flat = torch.empty(0, len(feature_cols))
            self.y = torch.empty(0)
            self.indices = np.array([], dtype=int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_flat[idx], self.y[idx]


class ReturnLSTMMLP(nn.Module):
    def __init__(self, n_flat_features: int, lstm_hidden: int = 30,
                 lstm_layers: int = 2, mlp_hidden: int = 128,
                 mlp_depth: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        concat_dim = lstm_hidden + n_flat_features
        layers = []
        in_dim = concat_dim
        h = mlp_hidden
        for i in range(mlp_depth):
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
            h = max(h // 2, 16)
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_seq, x_flat):
        _, (h, _) = self.lstm(x_seq)
        lstm_out = h[-1]
        combined = torch.cat([lstm_out, x_flat], dim=1)
        return self.mlp(combined).squeeze(-1)


class LSTMMLPReturnModel(ReturnModel):
    def __init__(self, lookback=12, lstm_hidden=30, lstm_layers=2,
                 mlp_hidden=128, mlp_depth=3, dropout=0.3, epochs=60,
                 patience=15, batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.mlp_hidden = mlp_hidden
        self.mlp_depth = mlp_depth
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_df, val_df, feature_cols):
        n_flat = len(feature_cols)
        train_ds = HybridDataset(train_df, feature_cols, self.lookback)
        val_ds = HybridDataset(val_df, feature_cols, self.lookback)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("LSTM+MLP: not enough sequential data, skipping")
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.model = ReturnLSTMMLP(
            n_flat, self.lstm_hidden, self.lstm_layers,
            self.mlp_hidden, self.mlp_depth, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        best_vl, best_w, no_imp = float("inf"), None, 0
        self.train_losses_ = []
        self.val_losses_ = []

        for _ in range(self.epochs):
            self.model.train()
            epoch_tl = []
            for X_seq, X_flat, y in train_dl:
                X_seq = X_seq.to(self.device)
                X_flat = X_flat.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                loss = loss_func(self.model(X_seq, X_flat), y)
                loss.backward()
                opt.step()
                epoch_tl.append(loss.item())
            self.train_losses_.append(np.mean(epoch_tl))

            self.model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.model(X_seq.to(self.device), X_flat.to(self.device)),
                                   y.to(self.device)).item()
                              for X_seq, X_flat, y in val_dl])
            self.val_losses_.append(vl)
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        return self

    def predict(self, df, feature_cols, history_df=None):
        if self.model is None:
            return np.full(len(df), np.nan)

        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds = HybridDataset(combined, feature_cols, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        dl = DataLoader(ds, batch_size=512)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_seq, X_flat, _ in dl:
                preds.append(self.model(X_seq.to(self.device),
                                        X_flat.to(self.device)).cpu().numpy())
        pred_arr = np.concatenate(preds)
        result_full = np.full(len(combined), np.nan)
        result_full[ds.indices] = pred_arr
        out = result_full[n_history:]
        nan_mask = np.isnan(out)
        if nan_mask.any() and (~nan_mask).any():
            out[nan_mask] = np.nanmedian(out)
        return out


class LSTMXGBReturnModel(ReturnModel):
    def __init__(self, lookback=12, lstm_hidden=16, lstm_layers=2,
                 lstm_dropout=0.2, lstm_lr=1e-3, lstm_epochs=40,
                 lstm_batch_size=256, lstm_patience=10,
                 loss_fn: str = "mse",
                 xgb_n_estimators=300, xgb_max_depth=6,
                 xgb_learning_rate=0.05, xgb_subsample=0.8):
        self.lookback = lookback
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_lr = lstm_lr
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.lstm_patience = lstm_patience
        self.loss_fn = loss_fn

        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_subsample = xgb_subsample

        self.lstm_model = None
        self.xgb_model = None
        self.emb_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_lstm(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        train_ds = ReturnsOnlyDataset(train_df, self.lookback)
        val_ds = ReturnsOnlyDataset(val_df, self.lookback)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("LSTM+XGB: not enough sequential data for LSTM stage")
            return

        train_dl = DataLoader(train_ds, batch_size=self.lstm_batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.lstm_model = ReturnLSTMReturnsOnly(
            self.lstm_hidden, self.lstm_layers, self.lstm_dropout
        ).to(self.device)

        opt = torch.optim.Adam(self.lstm_model.parameters(),
                               lr=self.lstm_lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)

        best_vl, best_w, no_imp = float("inf"), None, 0
        for _ in range(self.lstm_epochs):
            self.lstm_model.train()
            for X_seq, y in train_dl:
                X_seq, y = X_seq.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss_func(self.lstm_model(X_seq), y).backward()
                opt.step()

            self.lstm_model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.lstm_model(X.to(self.device)),
                              y.to(self.device)).item() for X, y in val_dl])
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone()
                          for k, v in self.lstm_model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.lstm_patience:
                break

        if best_w is not None:
            self.lstm_model.load_state_dict(best_w)
            self.lstm_model.to(self.device)

    def _extract_embeddings(self, df: pd.DataFrame, feature_cols: List[str],
                            history_df: Optional[pd.DataFrame] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.lstm_model is None:
            return np.array([]), np.array([]), np.array([])

        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds_ret = ReturnsOnlyDataset(combined, self.lookback)
        ds_hybrid = HybridDataset(combined, feature_cols, self.lookback)

        if len(ds_ret) == 0:
            return np.array([]), np.array([]), np.array([])

        dl_ret = DataLoader(ds_ret, batch_size=512, shuffle=False)

        self.lstm_model.eval()
        embeddings, targets_ret = [], []
        with torch.no_grad():
            for X_seq, y in dl_ret:
                _, (h, _) = self.lstm_model.lstm(X_seq.to(self.device))
                embeddings.append(h[-1].cpu().numpy())
                targets_ret.append(y.numpy())

        embeddings = np.vstack(embeddings)
        targets_ret = np.concatenate(targets_ret)

        if len(ds_hybrid) > 0:
            ret_indices = set(ds_ret.indices.tolist())
            hybrid_indices = set(ds_hybrid.indices.tolist())
            common = sorted(ret_indices & hybrid_indices)

            r2p = {idx: pos for pos, idx in enumerate(ds_ret.indices.tolist())}
            h2p = {idx: pos for pos, idx in enumerate(ds_hybrid.indices.tolist())}
            rp = [r2p[i] for i in common]
            hp = [h2p[i] for i in common]

            embeddings = embeddings[rp]
            targets_ret = targets_ret[rp]
            flat_features = ds_hybrid.X_flat[hp].numpy()
            final_indices = np.array(common)
        else:
            flat_features = np.zeros((len(embeddings), len(feature_cols)))
            final_indices = ds_ret.indices

        mask = final_indices >= n_history
        return embeddings[mask], flat_features[mask], targets_ret[mask]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feature_cols: List[str]) -> "LSTMXGBReturnModel":
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler

        self._train_lstm(train_df, val_df)
        if self.lstm_model is None:
            return self

        emb_tr, flat_tr, y_tr = self._extract_embeddings(train_df, feature_cols)
        emb_vl, flat_vl, y_vl = self._extract_embeddings(val_df, feature_cols)

        if len(emb_tr) == 0:
            print("LSTM+XGB: no embeddings extracted")
            return self

        self.emb_scaler = StandardScaler()
        emb_tr = self.emb_scaler.fit_transform(emb_tr)
        emb_vl = self.emb_scaler.transform(emb_vl) if len(emb_vl) > 0 else emb_vl

        X_tr = np.nan_to_num(np.hstack([emb_tr, flat_tr]), nan=0., posinf=0., neginf=0.)
        X_vl = np.nan_to_num(np.hstack([emb_vl, flat_vl]), nan=0., posinf=0., neginf=0.)

        use_gpu = torch.cuda.is_available()
        self.xgb_model = XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            early_stopping_rounds=20,
        )

        eval_set = [(X_vl, y_vl)] if len(X_vl) > 0 else None
        self.xgb_model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)

        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str],
                history_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.lstm_model is None or self.xgb_model is None or self.emb_scaler is None:
            return np.full(len(df), np.nan)

        emb, flat, _ = self._extract_embeddings(df, feature_cols, history_df)

        if len(emb) == 0:
            return np.full(len(df), np.nan)

        emb = self.emb_scaler.transform(emb)
        X = np.nan_to_num(np.hstack([emb, flat]), nan=0., posinf=0., neginf=0.)
        preds = self.xgb_model.predict(X)

        if len(preds) < len(df):
            result = np.full(len(df), np.nan)
            result[-len(preds):] = preds
            nan_mask = np.isnan(result)
            if nan_mask.any() and (~nan_mask).any():
                result[nan_mask] = np.nanmedian(result)
            return result
        return preds


class ReturnsOnlyDataset(Dataset):

    def __init__(self, df: pd.DataFrame, lookback: int = 12):
        self.lookback = lookback
        
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        
        ret_sequences = []
        targets = []
        indices = []
        
        for asset, grp in df.groupby("asset"):
            if len(grp) < lookback + 1:
                continue

            rets = grp["target"].values.astype(np.float32)
            idxs = grp.index.values

            for i in range(lookback, len(grp)):
                # rolling z-score over past lookback only
                past_rets = rets[i - lookback:i]
                ret_mean = past_rets.mean()
                ret_std = past_rets.std()
                if ret_std < 1e-8:
                    ret_std = 1.0
                past_z_rets = (past_rets - ret_mean) / ret_std

                ret_sequences.append(past_z_rets.reshape(-1, 1))
                targets.append(rets[i])
                indices.append(idxs[i])
        
        if ret_sequences:
            self.X_seq = torch.from_numpy(np.stack(ret_sequences))
            self.y = torch.from_numpy(np.array(targets))
            self.indices = np.array(indices)
        else:
            self.X_seq = torch.empty(0, lookback, 1)
            self.y = torch.empty(0)
            self.indices = np.array([], dtype=int)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]


class ReturnLSTMReturnsOnly(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x_seq):
        _, (h, _) = self.lstm(x_seq)
        return self.head(h[-1]).squeeze(-1)


class LSTMReturnsOnlyModel(ReturnModel):
    def __init__(self, lookback=12, hidden_size=32, num_layers=2,
                 dropout=0.2, epochs=60, patience=15,
                 batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            feature_cols: List[str]) -> "LSTMReturnsOnlyModel":
        train_ds = ReturnsOnlyDataset(train_df, self.lookback)
        val_ds = ReturnsOnlyDataset(val_df, self.lookback)
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            print("LSTMReturnsOnly: not enough sequential data")
            return self
        
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)
        
        self.model = ReturnLSTMReturnsOnly(
            self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        
        best_vl, best_w, no_imp = float("inf"), None, 0
        
        for epoch in range(self.epochs):
            self.model.train()
            for X_seq, y in train_dl:
                X_seq = X_seq.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                pred = self.model(X_seq)
                loss = loss_func(pred, y)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for X_seq, y in val_dl:
                    pred = self.model(X_seq.to(self.device))
                    vl = F.mse_loss(pred, y.to(self.device)).item()
                    val_losses.append(vl)
                vl = np.mean(val_losses)
            
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break
        
        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        
        return self
    
    def predict(self, df: pd.DataFrame, feature_cols: List[str],
                history_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.model is None:
            return np.full(len(df), np.nan)
        
        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0
        
        ds = ReturnsOnlyDataset(combined, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_seq, _ in dl:
                pred = self.model(X_seq.to(self.device)).cpu().numpy()
                preds.append(pred)
        
        preds = np.concatenate(preds)

        df_mask = ds.indices >= n_history
        preds = preds[df_mask]
        
        if len(preds) < len(df):
            result = np.full(len(df), np.nan)
            result[-len(preds):] = preds
            nan_mask = np.isnan(result)
            if nan_mask.any() and (~nan_mask).any():
                result[nan_mask] = np.nanmedian(result)
            return result
        return preds


class ReturnGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 16),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class GRUReturnModel(ReturnModel):
    def __init__(self, lookback=6, hidden_size=64, num_layers=1,
                 dropout=0.3, epochs=60, patience=15,
                 batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_df, val_df, feature_cols):
        n_features = len(feature_cols)
        train_ds = MonthlyDataset(train_df, feature_cols, self.lookback)
        val_ds = MonthlyDataset(val_df, feature_cols, self.lookback)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("GRU: not enough sequential data, skipping")
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.model = ReturnGRU(n_features, self.hidden_size,
                               self.num_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        best_vl, best_w, no_imp = float("inf"), None, 0
        self.train_losses_ = []
        self.val_losses_ = []

        for _ in range(self.epochs):
            self.model.train()
            epoch_tl = []
            for X, y in train_dl:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_func(self.model(X), y)
                loss.backward()
                opt.step()
                epoch_tl.append(loss.item())
            self.train_losses_.append(np.mean(epoch_tl))

            self.model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.model(X.to(self.device)),
                                   y.to(self.device)).item() for X, y in val_dl])
            self.val_losses_.append(vl)
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        return self

    def predict(self, df, feature_cols, history_df=None):
        if self.model is None:
            return np.full(len(df), np.nan)

        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds = MonthlyDataset(combined, feature_cols, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        dl = DataLoader(ds, batch_size=512)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in dl:
                preds.append(self.model(X.to(self.device)).cpu().numpy())
        pred_arr = np.concatenate(preds)
        result_full = np.full(len(combined), np.nan)
        result_full[ds.indices] = pred_arr
        out = result_full[n_history:]
        nan_mask = np.isnan(out)
        if nan_mask.any() and (~nan_mask).any():
            out[nan_mask] = np.nanmedian(out)
        return out



class ReturnGRUReturnsOnly(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size, 16),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x_seq):
        _, h = self.gru(x_seq)
        return self.head(h[-1]).squeeze(-1)


class GRUReturnsOnlyModel(ReturnModel):
    def __init__(self, lookback=12, hidden_size=32, num_layers=2,
                 dropout=0.2, epochs=60, patience=15,
                 batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_df, val_df, feature_cols):
        train_ds = ReturnsOnlyDataset(train_df, self.lookback)
        val_ds = ReturnsOnlyDataset(val_df, self.lookback)
        if len(train_ds) == 0 or len(val_ds) == 0:
            print("GRUReturnsOnly: not enough sequential data")
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.model = ReturnGRUReturnsOnly(
            self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        best_vl, best_w, no_imp = float("inf"), None, 0

        for _ in range(self.epochs):
            self.model.train()
            for X_seq, y in train_dl:
                X_seq, y = X_seq.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss_func(self.model(X_seq), y).backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.model(X.to(self.device)),
                              y.to(self.device)).item() for X, y in val_dl])
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        return self

    def predict(self, df, feature_cols, history_df=None):
        if self.model is None:
            return np.full(len(df), np.nan)
        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds = ReturnsOnlyDataset(combined, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        dl = DataLoader(ds, batch_size=512, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_seq, _ in dl:
                preds.append(self.model(X_seq.to(self.device)).cpu().numpy())
        preds = np.concatenate(preds)
        df_mask = ds.indices >= n_history
        preds = preds[df_mask]
        if len(preds) < len(df):
            result = np.full(len(df), np.nan)
            result[-len(preds):] = preds
            nan_mask = np.isnan(result)
            if nan_mask.any() and (~nan_mask).any():
                result[nan_mask] = np.nanmedian(result)
            return result
        return preds



class LinearReturnModel(ReturnModel):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, train_df, val_df, feature_cols):
        from sklearn.linear_model import Ridge
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["target"].values
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, df, feature_cols, **kwargs):
        if self.model is None:
            return np.full(len(df), np.nan)
        X = df[feature_cols].fillna(0).values
        return self.model.predict(X)



class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ReturnTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, 16),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


class TransformerReturnModel(ReturnModel):
    def __init__(self, lookback=6, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128,
                 dropout=0.3, epochs=60, patience=15,
                 batch_size=256, lr=1e-3, loss_fn: str = "mse"):
        self.lookback = lookback
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_df, val_df, feature_cols):
        n_features = len(feature_cols)
        train_ds = MonthlyDataset(train_df, feature_cols, self.lookback)
        val_ds = MonthlyDataset(val_df, feature_cols, self.lookback)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("Transformer: not enough sequential data, skipping")
            return self

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=512)

        self.model = ReturnTransformer(
            n_features, self.d_model, self.nhead, self.num_layers,
            self.dim_feedforward, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_func = LOSS_FUNCTIONS.get(self.loss_fn, mse_loss)
        best_vl, best_w, no_imp = float("inf"), None, 0
        self.train_losses_ = []
        self.val_losses_ = []

        for _ in range(self.epochs):
            self.model.train()
            epoch_tl = []
            for X, y in train_dl:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_func(self.model(X), y)
                loss.backward()
                opt.step()
                epoch_tl.append(loss.item())
            self.train_losses_.append(np.mean(epoch_tl))

            self.model.eval()
            with torch.no_grad():
                vl = np.mean([F.mse_loss(self.model(X.to(self.device)),
                                   y.to(self.device)).item() for X, y in val_dl])
            self.val_losses_.append(vl)
            if vl < best_vl:
                best_vl = vl
                best_w = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_w is not None:
            self.model.load_state_dict(best_w)
            self.model.to(self.device)
        return self

    def predict(self, df, feature_cols, history_df=None):
        if self.model is None:
            return np.full(len(df), np.nan)

        if history_df is not None:
            combined = pd.concat([history_df, df]).reset_index(drop=True)
            n_history = len(history_df)
        else:
            combined = df.reset_index(drop=True)
            n_history = 0

        ds = MonthlyDataset(combined, feature_cols, self.lookback)
        if len(ds) == 0:
            return np.full(len(df), np.nan)
        dl = DataLoader(ds, batch_size=512)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in dl:
                preds.append(self.model(X.to(self.device)).cpu().numpy())
        pred_arr = np.concatenate(preds)
        result_full = np.full(len(combined), np.nan)
        result_full[ds.indices] = pred_arr
        out = result_full[n_history:]
        nan_mask = np.isnan(out)
        if nan_mask.any() and (~nan_mask).any():
            out[nan_mask] = np.nanmedian(out)
        return out



class RFReturnModel(ReturnModel):
    def __init__(self, n_estimators=500, max_depth=20,
                 min_samples_leaf=30, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = None

    def fit(self, train_df, val_df, feature_cols):
        from cuml.ensemble import RandomForestRegressor as cuRF
        self.model = cuRF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["target"].values.astype(np.float32)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, df, feature_cols):
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        return np.array(self.model.predict(X))



class XGBReturnModel(ReturnModel):
    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 early_stopping_rounds=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, train_df, val_df, feature_cols):
        from xgboost import XGBRegressor
        import torch
        use_gpu = torch.cuda.is_available()
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            verbosity=0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            early_stopping_rounds=self.early_stopping_rounds,
        )
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["target"].values
        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df["target"].values
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return self

    def predict(self, df, feature_cols):
        X = df[feature_cols].fillna(0).values
        return self.model.predict(X)



def purged_kfold_split(df: pd.DataFrame, n_splits: int = 5, 
                       gap_months: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    months = sorted(df["date"].unique())
    n_months = len(months)
    fold_size = n_months // n_splits
    
    folds = []
    for i in range(n_splits):
        val_start_idx = i * fold_size
        val_end_idx = min(val_start_idx + fold_size, n_months)
        train_end_idx = max(0, val_start_idx - gap_months)
        
        if train_end_idx <= 0:
            continue
        
        train_months = months[:train_end_idx]
        val_months = months[val_start_idx:val_end_idx]
        
        train_fold = df[df["date"].isin(train_months)].copy()
        val_fold = df[df["date"].isin(val_months)].copy()
        
        if len(train_fold) > 0 and len(val_fold) > 0:
            folds.append((train_fold, val_fold))
    
    return folds



DEFAULT_SEARCH_SPACES = {
    "lstm": {
        "lookback": ("categorical", [3, 6, 9]),
        "hidden_size": ("categorical", [32, 64, 128]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "lstm_ts": {
        "lookback": ("categorical", [6, 9, 12, 18]),
        "hidden_size": ("categorical", [64, 128, 256]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "lstm_mlp": {
        "lookback": ("categorical", [6, 9, 12]),
        "lstm_hidden": ("fixed", 30),
        "lstm_layers": ("categorical", [1, 2]),
        "mlp_hidden": ("categorical", [64, 128, 256]),
        "mlp_depth": ("categorical", [2, 3, 4]),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "lstm_xgb": {
        "lookback": ("categorical", [6, 9, 12, 18, 24]),
        "lstm_hidden": ("categorical", [8, 16, 32]),
        "lstm_layers": ("categorical", [1, 2]),
        "lstm_dropout": ("float", 0.1, 0.4),
        "lstm_lr": ("float_log", 1e-4, 1e-2),
        "lstm_epochs": ("int_step", 20, 60, 10),
        "lstm_batch_size": ("categorical", [256, 512]),
        "lstm_patience": ("fixed", 10),
        "xgb_n_estimators": ("int_step", 100, 500, 100),
        "xgb_max_depth": ("int", 3, 8),
        "xgb_learning_rate": ("float_log", 0.01, 0.2),
        "xgb_subsample": ("float", 0.6, 1.0),
    },
    "lstm_ret": {
        "lookback": ("categorical", [6, 9, 12, 18, 24]),
        "hidden_size": ("categorical", [16, 32, 64]),
        "num_layers": ("categorical", [1, 2, 3]),
        "dropout": ("float", 0.1, 0.4),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512]),
    },
    "gru": {
        "lookback": ("categorical", [6, 9, 12, 18]),
        "hidden_size": ("categorical", [64, 128, 256]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "gru_ts": {
        "lookback": ("categorical", [6, 9, 12, 18]),
        "hidden_size": ("categorical", [64, 128, 256]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "transformer": {
        "lookback": ("categorical", [6, 9, 12, 18]),
        "d_model": ("categorical", [32, 64, 128]),
        "nhead": ("categorical", [2, 4, 8]),
        "num_layers": ("int", 1, 3),
        "dim_feedforward": ("categorical", [64, 128, 256]),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "transformer_ts": {
        "lookback": ("categorical", [6, 9, 12, 18]),
        "d_model": ("categorical", [32, 64, 128]),
        "nhead": ("categorical", [2, 4, 8]),
        "num_layers": ("int", 1, 3),
        "dim_feedforward": ("categorical", [64, 128, 256]),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512, 1024]),
    },
    "gru_ret": {
        "lookback": ("categorical", [6, 9, 12, 18, 24]),
        "hidden_size": ("categorical", [16, 32, 64]),
        "num_layers": ("categorical", [1, 2, 3]),
        "dropout": ("float", 0.1, 0.4),
        "lr": ("float_log", 1e-4, 1e-2),
        "epochs": ("int_step", 40, 100, 10),
        "patience": ("fixed", 15),
        "batch_size": ("categorical", [256, 512]),
    },
    "linreg": {
        "alpha": ("float_log", 0.01, 100.0),
    },
    "rf": {
        "n_estimators": ("int_step", 100, 300, 50),
        "max_depth": ("int", 5, 15),
        "min_samples_leaf": ("int", 10, 50),
    },
    "xgb": {
        "n_estimators": ("int_step", 100, 800, 100),
        "max_depth": ("int", 3, 8),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
}


def _suggest_param(trial, name: str, spec: tuple):
    kind = spec[0]
    if kind == "fixed":
        return spec[1]
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    elif kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif kind == "int_step":
        return trial.suggest_int(name, spec[1], spec[2], step=spec[3])
    elif kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    else:
        raise ValueError(f"Unknown param spec type: {kind}")


def optuna_search(
    model_class: str, 
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    feature_cols: List[str], 
    n_trials: int = 50,
    metric: str = "ic",
    loss_fn: str = "mse",
    use_cv: bool = False,
    cv_folds: int = 3,
    cv_gap_months: int = 1,
    search_spaces: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, Any], float]:

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.preprocessing import StandardScaler
    _clean = lambda x: x.replace([np.inf, -np.inf], np.nan).fillna(0)

    if use_cv:
        full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        cv_folds_list = purged_kfold_split(full_df, n_splits=cv_folds, gap_months=cv_gap_months)
        if len(cv_folds_list) == 0:
            print("Warning: CV produced no valid folds, falling back to single split")
            use_cv = False
    
    if not use_cv:
        scaler = StandardScaler()
        train_sc = train_df.copy()
        val_sc = val_df.copy()
        train_sc[feature_cols] = scaler.fit_transform(_clean(train_sc[feature_cols]))
        val_sc[feature_cols] = scaler.transform(_clean(val_sc[feature_cols]))

    spaces = {k: dict(v) for k, v in DEFAULT_SEARCH_SPACES.items()}
    if search_spaces:
        for mc, sp in search_spaces.items():
            if mc in spaces:
                spaces[mc].update(sp)
            else:
                spaces[mc] = dict(sp)

    if model_class not in spaces:
        raise ValueError(f"Unknown model_class: {model_class}. Available: {list(spaces.keys())}")

    model_space = spaces[model_class]

    MODEL_CLASS_MAP = {
        "lstm": LSTMReturnModel,
        "lstm_ts": LSTMReturnModel,
        "lstm_mlp": LSTMMLPReturnModel,
        "lstm_xgb": LSTMXGBReturnModel,
        "lstm_ret": LSTMReturnsOnlyModel,
        "gru": GRUReturnModel,
        "gru_ts": GRUReturnModel,
        "gru_ret": GRUReturnsOnlyModel,
        "transformer": TransformerReturnModel,
        "transformer_ts": TransformerReturnModel,
        "rf": RFReturnModel,
        "xgb": XGBReturnModel,
        "linreg": LinearReturnModel,
    }

    nn_models = ("lstm", "lstm_ts", "lstm_mlp", "lstm_ret", "lstm_xgb",
                 "gru", "gru_ts", "gru_ret", "transformer", "transformer_ts")
    seq_models = ("lstm", "lstm_ts", "lstm_mlp", "lstm_xgb", "lstm_ret",
                  "gru", "gru_ts", "gru_ret", "transformer", "transformer_ts")

    def objective(trial):
        params = {}
        for pname, spec in model_space.items():
            params[pname] = _suggest_param(trial, pname, spec)

        # Transformer: enforce d_model % nhead == 0
        if model_class == "transformer" and "d_model" in params and "nhead" in params:
            if params["d_model"] % params["nhead"] != 0:
                nhead_choices = model_space.get("nhead", ("categorical", [2, 4, 8]))[1]
                params["nhead"] = min(h for h in nhead_choices if params["d_model"] % h == 0)

        if model_class in nn_models and "loss_fn" not in params:
            params["loss_fn"] = loss_fn

        cls = MODEL_CLASS_MAP[model_class]
        model_fn = lambda: cls(**params)

        if use_cv:
            fold_metrics = []
            for train_fold, val_fold in cv_folds_list:
                scaler = StandardScaler()
                tr_sc = train_fold.copy()
                vl_sc = val_fold.copy()
                tr_sc[feature_cols] = scaler.fit_transform(_clean(tr_sc[feature_cols]))
                vl_sc[feature_cols] = scaler.transform(_clean(vl_sc[feature_cols]))
                
                model = model_fn()
                model.fit(tr_sc, vl_sc, feature_cols)
                
                if model_class in seq_models:
                    preds = model.predict(vl_sc, feature_cols, history_df=tr_sc)
                else:
                    preds = model.predict(vl_sc, feature_cols)
                
                vl_result = vl_sc.copy()
                vl_result["pred"] = preds
                fold_metric = compute_metric(vl_result, metric=metric)
                fold_metrics.append(fold_metric)
            
            return np.mean(fold_metrics)
        else:
            model = model_fn()
            model.fit(train_sc, val_sc, feature_cols)
            
            if model_class in seq_models:
                preds = model.predict(val_sc, feature_cols, history_df=train_sc)
            else:
                preds = model.predict(val_sc, feature_cols)
            
            val_result = val_sc.copy()
            val_result["pred"] = preds
            return compute_metric(val_result, metric=metric)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    metric_name = metric.upper()
    print(f"\n{model_class.upper()} best {metric_name}: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params, study.best_value
