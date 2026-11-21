# %% [markdown]
# # Learning from Expert Disagreement: Reduced-Montage EEG Classification of Harmful Brain Activity with Patient-Level Conformal Uncertainty
#
# Kaggle notebook that trains and evaluates every model in the manuscript.
#
# **Setup on Kaggle**
# 1. New notebook -> *Add Input* -> competition **HMS - Harmful Brain Activity Classification** (accept the rules first).
# 2. Settings -> Accelerator **GPU T4 x2** or **P100**. Internet can stay off.
# 3. Set "RUN" (cell 2) to the experiments for this session. One experiment = 5 folds, about 40-70 min on one GPU.
#    Spread the list over several sessions if needed; every experiment saves its own out-of-fold file,
#    and the last cell builds all paper tables from whatever files exist.
# 4. Outputs in "/kaggle/working/results/" are the numbers for the manuscript tables (Table 2-6, Fig. 2-4).

# %%
import os, gc, json, math, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:          # analysis-only environments
    HAS_TORCH = False

# %%
CFG = dict(
    data_dir="/kaggle/input/hms-harmful-brain-activity-classification",
    out_dir="/kaggle/working",
    fs_in=200, fs=100, win_sec=50,          # 50 s window, labelled centre 10 s
    n_folds=5, inner_val_frac=0.10,
    epochs=15, batch_size=64, lr=1e-3, wd=1e-2, seed=42, num_workers=2,
    d_model=128, n_layers=2, n_heads=4, dropout=0.1,
    alpha=0.10,                 # conformal miscoverage -> 90% target coverage
    plausible_tau=0.25,         # a class is "expert-plausible" if it got >= 25% of votes
    high_consensus_votes=10,    # high-consensus test subset
    n_group_draws=20,           # random one-per-patient draws for patient-level conformal
    n_boot=1000,
    folds_to_run=None,          # e.g. [0] for a quick check
    max_eegs=None,              # e.g. 2000 for a smoke test
)

if not Path(CFG["data_dir"]).exists():          # find the competition data wherever Kaggle mounted it
    cands = [str(q.parent) for q in Path("/kaggle/input").rglob("train.csv")] if Path("/kaggle/input").exists() else []
    print("Data dir not found; candidates:", cands)
    if cands:
        CFG["data_dir"] = cands[0]
print("DATA:", CFG["data_dir"], "| exists:", Path(CFG["data_dir"], "train.csv").exists())

EXPERIMENTS = {
    # RQ1: loss ablation (full montage)
    "full18_kl":        dict(arch="tfm", montage="full18", loss="kl"),
    "full18_ce":        dict(arch="tfm", montage="full18", loss="ce"),
    "eegnet_full18_kl": dict(arch="eegnet", montage="full18", loss="kl"),
    # RQ2: fixed reduced montages
    "temporal8_kl":       dict(arch="tfm", montage="temporal8", loss="kl"),
    "parasagittal8_kl":   dict(arch="tfm", montage="parasagittal8", loss="kl"),
    "frontotemporal4_kl": dict(arch="tfm", montage="frontotemporal4", loss="kl"),
    "frontal2_kl":        dict(arch="tfm", montage="frontal2", loss="kl"),
    # RQ2: learned channel subsets (concrete selector)
    "learned_k2_kl":  dict(arch="tfm", k=2,  loss="kl"),
    "learned_k4_kl":  dict(arch="tfm", k=4,  loss="kl"),
    "learned_k6_kl":  dict(arch="tfm", k=6,  loss="kl"),
    "learned_k8_kl":  dict(arch="tfm", k=8,  loss="kl"),
    "learned_k10_kl": dict(arch="tfm", k=10, loss="kl"),
}
# Experiments to train in THIS session (edit per session).
RUN = ["full18_kl", "full18_ce", "temporal8_kl", "learned_k4_kl", "learned_k8_kl",
       "eegnet_full18_kl", "parasagittal8_kl", "frontotemporal4_kl", "frontal2_kl",
       "learned_k2_kl", "learned_k6_kl", "learned_k10_kl"]   # all 12 experiments (~6 h on one T4)

# Quick test: set QUICK = True in a cell before this one (1 fold, small sample, 2 epochs).
QUICK = globals().get("QUICK", False)
if QUICK:
    CFG.update(max_eegs=1500, folds_to_run=[0], epochs=2, n_boot=100)
    RUN = ["full18_kl", "learned_k4_kl"]
CACHE_DIR = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
OUT = Path(CFG["out_dir"]); (OUT / "results").mkdir(parents=True, exist_ok=True); (OUT / "models").mkdir(exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s)
    if HAS_TORCH:
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])

# %% [markdown]
# ## 1. Labels, montages and one labelled window per EEG

# %%
VOTES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
CLASSES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
EEG_COLS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
            "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]
BIPOLAR = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),     # 0-3  left temporal (LL)
           ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),     # 4-7  right temporal (RL)
           ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),     # 8-11 left parasagittal (LP)
           ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),     # 12-15 right parasagittal (RP)
           ("Fz", "Cz"), ("Cz", "Pz")]                                  # 16-17 midline
BIP_NAMES = [f"{a}-{b}" for a, b in BIPOLAR]
N_BIP = len(BIPOLAR)
MONTAGES = {
    "full18": list(range(18)),
    "temporal8": [0, 1, 2, 3, 4, 5, 6, 7],          # hairline/temporal chains (10 electrodes)
    "parasagittal8": [8, 9, 10, 11, 12, 13, 14, 15],
    "frontotemporal4": [0, 1, 4, 5],                # Fp1-F7, F7-T3, Fp2-F8, F8-T4
    "frontal2": [0, 4],                             # Fp1-F7, Fp2-F8
}
HEMI_FLIP = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17]  # left<->right


def build_meta(train):
    """One labelled 50-s window per EEG recording: the middle annotated sub-segment.
    Using one window per recording avoids near-duplicate overlapping windows."""
    train = train.sort_values(["eeg_id", "eeg_label_offset_seconds"]).copy()
    train["rank"] = train.groupby("eeg_id").cumcount()
    train["n_sub"] = train.groupby("eeg_id")["eeg_id"].transform("size")
    meta = train[train["rank"] == train["n_sub"] // 2].reset_index(drop=True)
    meta["total_votes"] = meta[VOTES].sum(1)
    meta = meta[meta["total_votes"] > 0].reset_index(drop=True)
    return meta


def vote_targets(meta):
    v = meta[VOTES].values.astype(np.float32)
    return v / v.sum(1, keepdims=True)

# %% [markdown]
# ## 2. Pre-processing: bipolar double banana, 60 Hz notch, 0.5-40 Hz band-pass, 100 Hz

# %%
SOS_BP = butter(4, [0.5, 40.0], btype="band", fs=CFG["fs_in"], output="sos")
NOTCH_B, NOTCH_A = iirnotch(60.0, 30.0, fs=CFG["fs_in"])
WIN_IN = CFG["win_sec"] * CFG["fs_in"]          # 10,000 samples at 200 Hz
WIN = CFG["win_sec"] * CFG["fs"]                # 5,000 samples at 100 Hz
CH_IDX = {c: i for i, c in enumerate(EEG_COLS)}


def preprocess_window(x):
    """x: (WIN_IN, 19) referential EEG in microvolts -> (18, WIN) float16 bipolar."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < WIN_IN:
        x = np.vstack([x, np.full((WIN_IN - len(x), x.shape[1]), np.nan)])
    col_mean = np.nanmean(x, 0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    r, c = np.where(np.isnan(x))
    x[r, c] = col_mean[c]
    b = np.stack([x[:, CH_IDX[a]] - x[:, CH_IDX[c2]] for a, c2 in BIPOLAR], 0)
    b = filtfilt(NOTCH_B, NOTCH_A, b, axis=1)
    b = sosfiltfilt(SOS_BP, b, axis=1)
    b = resample_poly(b, CFG["fs"], CFG["fs_in"], axis=1)
    b = np.clip(b, -500.0, 500.0) / 50.0            # robust fixed scaling
    return b.astype(np.float16)


def load_eeg_window(eeg_id, offset_s, split="train"):
    df = pd.read_parquet(f"{CFG['data_dir']}/{split}_eegs/{eeg_id}.parquet", columns=EEG_COLS)
    s = int(round(offset_s * CFG["fs_in"]))
    return preprocess_window(df.values[s:s + WIN_IN])


def build_cache(meta, path):
    path = Path(path)
    if path.exists():
        X = np.load(path, mmap_mode="r")
        if X.shape[0] == len(meta):
            return X
    from joblib import Parallel, delayed
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(meta), N_BIP, WIN))
    chunk = 512
    for s in range(0, len(meta), chunk):
        rows = meta.iloc[s:s + chunk]
        arrs = Parallel(n_jobs=4)(delayed(load_eeg_window)(e, o)
                                  for e, o in zip(rows.eeg_id, rows.eeg_label_offset_seconds))
        X[s:s + len(arrs)] = np.stack(arrs)
        print(f"cached {s + len(arrs)}/{len(meta)}", end="\r")
    X.flush(); del X
    return np.load(path, mmap_mode="r")

# %% [markdown]
# ## 3. Models: shared per-channel CNN + Transformer, optional concrete channel selector; EEGNet baseline

# %%
if HAS_TORCH:
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ConcreteSelector(nn.Module):
        """Learns k of n input channels end-to-end (concrete / Gumbel-softmax selection)."""
        def __init__(self, n_in, k):
            super().__init__()
            self.logits = nn.Parameter(torch.randn(k, n_in) * 0.01)
            self.temp = 10.0

        def forward(self, x):
            if self.training:
                u = torch.rand_like(self.logits).clamp(1e-10, 1 - 1e-10)
                g = -torch.log(-torch.log(u))
                S = F.softmax((self.logits + g) / self.temp, dim=-1)
            else:
                S = F.one_hot(self.logits.argmax(-1), self.logits.shape[1]).float()
            return torch.einsum("kc,bct->bkt", S, x), S

        def duplicate_penalty(self):
            P = F.softmax(self.logits / self.temp, dim=-1)        # (k, n)
            return F.relu(P.sum(0) - 1.0).sum()

        def selected(self):
            return self.logits.argmax(-1).tolist()

    class ChannelEncoder(nn.Module):
        def __init__(self, d):
            super().__init__()
            ch, pools, layers = [1, 32, 64, 96, d], [4, 4, 4, 2], []
            for i in range(4):
                layers += [nn.Conv1d(ch[i], ch[i + 1], 7, padding=3, bias=False),
                           nn.BatchNorm1d(ch[i + 1]), nn.GELU(), nn.MaxPool1d(pools[i])]
            self.net = nn.Sequential(*layers)          # 5000 -> 39 time steps

        def forward(self, x):
            return self.net(x)

    class IIICTransformer(nn.Module):
        def __init__(self, montage=None, k=None, d=128, n_layers=2, n_heads=4, dropout=0.1, n_cls=6):
            super().__init__()
            self.sel = ConcreteSelector(N_BIP, k) if k else None
            idx = montage if montage is not None else list(range(N_BIP))
            self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
            self.enc = ChannelEncoder(d)
            self.ch_emb = nn.Parameter(torch.randn(N_BIP, d) * 0.02)
            self.time_emb = nn.Parameter(torch.randn(1, 1, 64, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_cls))

        def forward(self, x):
            if self.sel is not None:
                x, S = self.sel(x)
            else:
                x = x[:, self.idx]
                S = F.one_hot(self.idx, N_BIP).float()
            B, K, T = x.shape
            h = self.enc(x.reshape(B * K, 1, T))                 # (B*K, d, T')
            Tp = h.shape[-1]
            h = h.transpose(1, 2).reshape(B, K, Tp, -1)
            h = h + (S @ self.ch_emb)[None, :, None, :] + self.time_emb[:, :, :Tp]
            h = self.tf(h.reshape(B, K * Tp, -1))
            return self.head(h.mean(1))

    class EEGNet(nn.Module):
        """Lawhern et al. (2018) EEGNet adapted to 100 Hz, 50-s windows."""
        def __init__(self, montage, n_cls=6, F1=16, D=2, F2=32, k1=64, drop=0.25):
            super().__init__()
            self.register_buffer("idx", torch.tensor(montage, dtype=torch.long))
            C = len(montage)
            self.net = nn.Sequential(
                nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop),
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop),
                nn.AdaptiveAvgPool2d((1, 16)), nn.Flatten(), nn.Linear(F2 * 16, n_cls))
            self.sel = None

        def forward(self, x):
            return self.net(x[:, self.idx].unsqueeze(1))

    def build_model(exp):
        montage = MONTAGES.get(exp.get("montage", "full18"))
        if exp["arch"] == "eegnet":
            return EEGNet(montage)
        return IIICTransformer(montage=None if exp.get("k") else montage, k=exp.get("k"),
                               d=CFG["d_model"], n_layers=CFG["n_layers"],
                               n_heads=CFG["n_heads"], dropout=CFG["dropout"])

    class EEGDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y, idx, train):
            self.X, self.Y, self.idx, self.train = X, Y, np.asarray(idx), train

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            x = np.asarray(self.X[j], dtype=np.float32)
            if self.train:
                if random.random() < 0.5:
                    x = x[HEMI_FLIP]                  # left-right swap keeps the class label
                if random.random() < 0.5:
                    x = -x                            # polarity inversion
                shift = random.randint(-CFG["fs"], CFG["fs"])  # +/- 1 s jitter
                x = np.roll(x, shift, axis=1)
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.Y[j])

    def loss_fn(logits, y, mode):
        if mode == "kl":
            return F.kl_div(F.log_softmax(logits.float(), -1), y, reduction="batchmean")
        return F.cross_entropy(logits.float(), y.argmax(-1))

    @torch.no_grad()
    def predict_logits(model, X, Y, idx, bs=128):
        model.eval()
        dl = torch.utils.data.DataLoader(EEGDataset(X, Y, idx, False), batch_size=bs,
                                         shuffle=False, num_workers=CFG["num_workers"])
        out = []
        for xb, _ in dl:
            with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
                out.append(model(xb.to(DEV)).float().cpu().numpy())
        return np.concatenate(out)

    def train_one(exp, X, Y, tr_idx, iv_idx, tag):
        seed_all(CFG["seed"])
        model = build_model(exp).to(DEV)
        dl = torch.utils.data.DataLoader(EEGDataset(X, Y, tr_idx, True), batch_size=CFG["batch_size"],
                                         shuffle=True, num_workers=CFG["num_workers"], drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
        E = CFG["epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=E * len(dl), pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
        best, best_state = float("inf"), None
        for ep in range(E):
            if model.sel is not None:
                model.sel.temp = 10.0 * (0.01 ** (ep / max(E - 1, 1)))   # 10 -> 0.1
            model.train(); t0, tot = time.time(), 0.0
            for xb, yb in dl:
                xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
                with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
                    logits = model(xb)
                loss = loss_fn(logits, yb, exp["loss"])
                if model.sel is not None:
                    loss = loss + 0.1 * model.sel.duplicate_penalty()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item()
            lv = predict_logits(model, X, Y, iv_idx)
            val_kl = kl_div_np(Y[iv_idx], softmax_np(lv))
            sel_txt = f" sel={[BIP_NAMES[c] for c in model.sel.selected()]}" if model.sel is not None else ""
            print(f"[{tag}] ep{ep+1:02d} train={tot/len(dl):.4f} innerKL={val_kl:.4f} {time.time()-t0:.0f}s{sel_txt}", flush=True)
            if model.sel is not None and ep < E // 2:
                continue                                   # selector still soft: do not checkpoint
            if val_kl < best:
                best = val_kl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model

# %% [markdown]
# ## 4. Metrics, calibration and conformal prediction (NumPy only)

# %%
def softmax_np(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def kl_div_np(y, p, eps=1e-15):
    """Mean KL(y || p): the competition metric."""
    p = np.clip(p, eps, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * (np.log(np.clip(y, eps, 1.0)) - np.log(p)), 0.0)
    return float(t.sum(1).mean())


def fit_temperature(logits, y):
    """Temperature that minimises KL to the expert-vote distribution (grid + refinement)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    vals = [kl_div_np(y, softmax_np(logits, T)) for T in grid]
    T0 = grid[int(np.argmin(vals))]
    fine = np.linspace(T0 * 0.9, T0 * 1.1, 101)
    return float(fine[int(np.argmin([kl_div_np(y, softmax_np(logits, T)) for T in fine]))])


def ece_np(p, lab, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == lab).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier_soft(p, y):
    return float(((p - y) ** 2).sum(1).mean())


def point_metrics(p, y):
    lab = y.argmax(1)
    pred = p.argmax(1)
    out = dict(KL=kl_div_np(y, p), ECE=ece_np(p, lab), Brier=brier_soft(p, y),
               BalAcc=balanced_accuracy_score(lab, pred),
               MacroF1=f1_score(lab, pred, average="macro"))
    try:
        out["MacroAUROC"] = roc_auc_score(lab, p, multi_class="ovr", average="macro", labels=list(range(6)))
    except ValueError:
        out["MacroAUROC"] = np.nan
    for c, name in enumerate(CLASSES):
        m = lab == c
        out[f"Sens_{name}"] = float((pred[m] == c).mean()) if m.any() else np.nan
    return out


def conformal_quantile(scores, alpha, weights=None):
    """Split-conformal threshold. Unweighted: ceil((n+1)(1-alpha))-th smallest score.
    Weighted: weighted quantile with an extra +inf point carrying the mean weight."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 1.0
    if weights is None:
        k = int(math.ceil((n + 1) * (1 - alpha)))
        return 1.0 if k > n else float(np.sort(s)[k - 1])
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(s)
    s, w = s[order], w[order]
    w_all = np.append(w, w.mean())
    cum = np.cumsum(w_all) / w_all.sum()
    i = int(np.searchsorted(cum, 1 - alpha))
    return 1.0 if i >= n else float(s[i])


def lac_scores(p, lab):
    return 1.0 - p[np.arange(len(p)), lab]


def make_sets(p, q):
    """q scalar or per-class vector; include class c if 1 - p_c <= q_c. Never return an empty set."""
    S = (1.0 - p) <= np.asarray(q)[None] if np.ndim(q) else (1.0 - p) <= q
    S = np.array(S, dtype=bool)
    empty = ~S.any(1)
    S[empty, p[empty].argmax(1)] = True
    return S


def set_metrics(S, y, patients, tau):
    lab = y.argmax(1)
    covered = S[np.arange(len(S)), lab]
    plausible = y >= tau
    plaus_cov = (S | ~plausible).all(1)                # every expert-plausible class is in the set
    size = S.sum(1)
    ent = -(np.where(y > 0, y * np.log(np.clip(y, 1e-12, 1)), 0)).sum(1)
    df = pd.DataFrame(dict(p=patients, c=covered))
    per_pat = df.groupby("p")["c"].mean()
    rho = spearmanr(size, ent).correlation if size.std() > 0 else np.nan
    return dict(Coverage=float(covered.mean()), PatientCoverage=float(per_pat.mean()),
                WorstDecilePatientCov=float(per_pat.quantile(0.10)),
                PlausibleSetCoverage=float(plaus_cov.mean()), MeanSetSize=float(size.mean()),
                SingletonRate=float((size == 1).mean()), Spearman_size_entropy=float(rho))


def conformal_all(p_cal, y_cal, pat_cal, p_te, y_te, pat_te, alpha, tau, n_draws, seed):
    lab_cal = y_cal.argmax(1)
    sc = lac_scores(p_cal, lab_cal)
    res = {}
    # 1) naive split conformal (assumes all calibration windows exchangeable)
    res["Naive"] = set_metrics(make_sets(p_te, conformal_quantile(sc, alpha)), y_te, pat_te, tau)
    # 2) class-conditional (Mondrian)
    qc = np.array([conformal_quantile(sc[lab_cal == c], alpha) for c in range(6)])
    res["Mondrian"] = set_metrics(make_sets(p_te, qc), y_te, pat_te, tau)
    # 3) patient-weighted: each patient carries equal total weight
    cnt = pd.Series(pat_cal).map(pd.Series(pat_cal).value_counts()).values
    res["PatientWeighted"] = set_metrics(make_sets(p_te, conformal_quantile(sc, alpha, 1.0 / cnt)),
                                         y_te, pat_te, tau)
    # 4) patient-level (one random window per calibration patient), averaged over draws
    rng = np.random.RandomState(seed)
    dfc = pd.DataFrame(dict(p=pat_cal, i=np.arange(len(pat_cal))))
    draws = []
    for _ in range(n_draws):
        pick = dfc.groupby("p")["i"].apply(lambda s: rng.choice(s.values)).values
        draws.append(set_metrics(make_sets(p_te, conformal_quantile(sc[pick], alpha)), y_te, pat_te, tau))
    res["PatientLevel"] = {k: float(np.nanmean([d[k] for d in draws])) for k in draws[0]}
    return res


def patient_bootstrap(p, y, patients, n_boot, seed, fn):
    rng = np.random.RandomState(seed)
    up = np.unique(patients)
    groups = {u: np.where(patients == u)[0] for u in up}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[u] for u in rng.choice(up, len(up), replace=True)])
        try:
            vals.append(fn(p[idx], y[idx]))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])

# %% [markdown]
# ## 5. Load data, build cache, patient-grouped folds

# %%
def split_calib_test(va_idx, patients, seed):
    """Split a validation fold by patient into calibration (50%) and test (50%) halves."""
    rng = np.random.RandomState(seed)
    up = np.unique(patients[va_idx]); rng.shuffle(up)
    cal_p = set(up[: len(up) // 2])
    is_cal = np.array([patients[i] in cal_p for i in va_idx])
    return va_idx[is_cal], va_idx[~is_cal]


def make_folds(meta):
    gkf = GroupKFold(n_splits=CFG["n_folds"])
    return list(gkf.split(np.zeros(len(meta)), groups=meta["patient_id"].values))


if __name__ == "__main__" and Path(CFG["data_dir"]).exists():
    train_csv = pd.read_csv(f"{CFG['data_dir']}/train.csv")
    meta = build_meta(train_csv)
    if CFG["max_eegs"]:
        meta = meta.sample(CFG["max_eegs"], random_state=CFG["seed"]).reset_index(drop=True)
    Y = vote_targets(meta)
    PAT = meta["patient_id"].values
    folds = make_folds(meta)
    summary = dict(train_rows=len(train_csv), eeg_recordings=len(meta), patients=int(meta.patient_id.nunique()),
                   class_counts=dict(zip(CLASSES, np.bincount(Y.argmax(1), minlength=6).tolist())),
                   median_votes=float(meta.total_votes.median()),
                   high_consensus_windows=int((meta.total_votes >= CFG["high_consensus_votes"]).sum()))
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(OUT / "results" / "table1_dataset.json", "w"), indent=2)
    X = build_cache(meta, CACHE_DIR / f"X_{len(meta)}.npy")
    meta.to_csv(OUT / "results" / "meta_used.csv", index=False)

# %% [markdown]
# ## 6. Train: 5-fold patient-grouped CV per experiment, save out-of-fold logits

# %%
def run_experiment(name):
    exp = EXPERIMENTS[name]
    N = len(meta)
    logits = np.full((N, 6), np.nan, dtype=np.float32)
    fold_id = np.full(N, -1); half = np.full(N, -1)        # half: 0 = calibration, 1 = test
    selected = {}
    for f, (tr, va) in enumerate(folds):
        if CFG["folds_to_run"] is not None and f not in CFG["folds_to_run"]:
            continue
        rng = np.random.RandomState(CFG["seed"] + f)
        tr_p = np.unique(PAT[tr]); rng.shuffle(tr_p)
        iv_p = set(tr_p[: max(1, int(len(tr_p) * CFG["inner_val_frac"]))])
        iv_mask = np.array([PAT[i] in iv_p for i in tr])
        model = train_one(exp, X, Y, tr[~iv_mask], tr[iv_mask], f"{name} f{f}")
        logits[va] = predict_logits(model, X, Y, va)
        cal, te = split_calib_test(va, PAT, CFG["seed"] + 100 + f)
        fold_id[va] = f; half[cal] = 0; half[te] = 1
        if getattr(model, "sel", None) is not None:
            selected[f] = [BIP_NAMES[c] for c in model.sel.selected()]
        torch.save(model.state_dict(), OUT / "models" / f"{name}_f{f}.pt")
        del model; gc.collect(); torch.cuda.empty_cache()
    np.savez(OUT / "results" / f"oof_{name}.npz", logits=logits, fold=fold_id, half=half,
             eeg_id=meta.eeg_id.values, selected=json.dumps(selected))
    print(name, "saved", selected)


if __name__ == "__main__" and HAS_TORCH and Path(CFG["data_dir"]).exists():
    for name in RUN:
        if (OUT / "results" / f"oof_{name}.npz").exists():
            print("skip (exists)", name); continue
        run_experiment(name)

# %% [markdown]
# ## 7. Analysis: builds every results table of the manuscript from the saved out-of-fold files

# %%
def analyze(name, meta, Y):
    d = np.load(OUT / "results" / f"oof_{name}.npz", allow_pickle=True)
    L, fold, half = d["logits"], d["fold"], d["half"]
    PAT = meta["patient_id"].values
    hc = meta["total_votes"].values >= CFG["high_consensus_votes"]
    rows, conf_rows = [], []
    p_all = np.full_like(L, np.nan, dtype=np.float64)
    for f in sorted(set(fold[fold >= 0])):
        cal = np.where((fold == f) & (half == 0))[0]
        te = np.where((fold == f) & (half == 1))[0]
        T = fit_temperature(L[cal], Y[cal])
        p_cal, p_te = softmax_np(L[cal], T), softmax_np(L[te], T)
        p_all[te] = p_te
        m = point_metrics(p_te, Y[te]); m.update(fold=f, T=T, subset="all")
        m_raw = point_metrics(softmax_np(L[te]), Y[te])
        m["KL_uncal"], m["ECE_uncal"] = m_raw["KL"], m_raw["ECE"]
        rows.append(m)
        te_hc = te[hc[te]]
        if len(te_hc) > 50:
            mh = point_metrics(softmax_np(L[te_hc], T), Y[te_hc]); mh.update(fold=f, T=T, subset="high_consensus")
            rows.append(mh)
        cr = conformal_all(p_cal, Y[cal], PAT[cal], p_te, Y[te], PAT[te], CFG["alpha"],
                           CFG["plausible_tau"], CFG["n_group_draws"], CFG["seed"] + f)
        for meth, v in cr.items():
            v = dict(v); v.update(fold=f, method=meth); conf_rows.append(v)
    te_all = np.where(half == 1)[0]
    ci_kl = patient_bootstrap(p_all[te_all], Y[te_all], PAT[te_all], CFG["n_boot"], CFG["seed"],
                              lambda p, y: kl_div_np(y, p))
    ci_auc = patient_bootstrap(p_all[te_all], Y[te_all], PAT[te_all], CFG["n_boot"], CFG["seed"],
                               lambda p, y: roc_auc_score(y.argmax(1), p, multi_class="ovr",
                                                          average="macro", labels=list(range(6))))
    sel = json.loads(str(d["selected"]))
    return (pd.DataFrame(rows), pd.DataFrame(conf_rows),
            dict(KL_CI=ci_kl.tolist(), AUROC_CI=ci_auc.tolist(), p_test=p_all, te=te_all), sel)


def paired_patient_bootstrap(p1, p2, y, patients, n_boot, seed, fn):
    """Delta = fn(p2) - fn(p1) on identical patient resamples. Returns mean, 95% CI, two-sided p."""
    rng = np.random.RandomState(seed)
    up = np.unique(patients)
    groups = {u: np.where(patients == u)[0] for u in up}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[u] for u in rng.choice(up, len(up), replace=True)])
        try:
            deltas.append(fn(p2[idx], y[idx]) - fn(p1[idx], y[idx]))
        except ValueError:
            pass
    d = np.asarray(deltas)
    pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(fn(p2, y) - fn(p1, y)), np.percentile(d, [2.5, 97.5]), float(pval)


def fmt(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def build_tables(meta, Y):
    names = [n for n in EXPERIMENTS if (OUT / "results" / f"oof_{n}.npz").exists()]
    point, confs, cis, sels = {}, {}, {}, {}
    for n in names:
        point[n], confs[n], cis[n], sels[n] = analyze(n, meta, Y)
    cols = ["KL", "MacroAUROC", "BalAcc", "MacroF1", "ECE", "Brier", "KL_uncal", "ECE_uncal"]
    t2 = []
    for n in names:
        a = point[n][point[n].subset == "all"]
        h = point[n][point[n].subset == "high_consensus"]
        r = {"Experiment": n, **{c: fmt(a[c]) for c in cols}}
        r["KL 95% CI"] = "[{:.3f}, {:.3f}]".format(*cis[n]["KL_CI"])
        r["AUROC 95% CI"] = "[{:.3f}, {:.3f}]".format(*cis[n]["AUROC_CI"])
        if len(h):
            r["KL (high-consensus)"] = fmt(h["KL"]); r["AUROC (high-consensus)"] = fmt(h["MacroAUROC"])
        t2.append(r)
    t2 = pd.DataFrame(t2); t2.to_csv(OUT / "results" / "table2_point_metrics.csv", index=False)
    t3 = pd.DataFrame([{"Experiment": n, **{f"Sens_{c}": fmt(point[n][point[n].subset == "all"][f"Sens_{c}"])
                                           for c in CLASSES}} for n in names])
    t3.to_csv(OUT / "results" / "table3_per_class_sensitivity.csv", index=False)
    t4 = []
    for n in names:
        for meth, g in confs[n].groupby("method"):
            t4.append({"Experiment": n, "Method": meth,
                       **{c: fmt(g[c]) for c in ["Coverage", "PatientCoverage", "WorstDecilePatientCov",
                                                  "PlausibleSetCoverage", "MeanSetSize", "SingletonRate",
                                                  "Spearman_size_entropy"]}})
    t4 = pd.DataFrame(t4); t4.to_csv(OUT / "results" / "table4_conformal.csv", index=False)
    t5 = []
    for n in names:
        if sels[n]:
            from collections import Counter
            cnt = Counter(c for v in sels[n].values() for c in v)
            t5.append({"Experiment": n, "Selected channels (count over folds)":
                       ", ".join(f"{c}×{k}" for c, k in cnt.most_common())})
    pd.DataFrame(t5).to_csv(OUT / "results" / "table5_selected_channels.csv", index=False)
    # paired patient-level bootstrap vs the full-montage KL model (pooled test halves)
    tests = []
    if "full18_kl" in point:
        PATv = meta["patient_id"].values
        te_ref, p_ref = cis["full18_kl"]["te"], cis["full18_kl"]["p_test"]
        auc = lambda p, y: roc_auc_score(y.argmax(1), p, multi_class="ovr", average="macro", labels=list(range(6)))
        metrics = {"KL": lambda p, y: kl_div_np(y, p), "MacroAUROC": auc,
                   "ECE": lambda p, y: ece_np(p, y.argmax(1))}
        for n in names:
            if n == "full18_kl" or not np.array_equal(cis[n]["te"], te_ref):
                continue
            for mname, fn in metrics.items():
                delta, ci, pv = paired_patient_bootstrap(p_ref[te_ref], cis[n]["p_test"][te_ref], Y[te_ref],
                                                         PATv[te_ref], CFG["n_boot"], CFG["seed"], fn)
                tests.append(dict(Comparison=f"{n} minus full18_kl", Metric=mname, Delta=round(delta, 4),
                                  CI_low=round(ci[0], 4), CI_high=round(ci[1], 4), p=pv))
    if tests:
        tt = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
        m = len(tt)
        tt["p_holm"] = np.minimum(1.0, np.maximum.accumulate([p * (m - i) for i, p in enumerate(tt.p)]))
        tt.to_csv(OUT / "results" / "table6_tests.csv", index=False)
    return t2, t4


def channel_occlusion(name, meta, Y, max_n=3000):
    """Fig. 4: drop in predicted probability of the expert-majority class when each bipolar channel is zeroed."""
    d = np.load(OUT / "results" / f"oof_{name}.npz", allow_pickle=True)
    fold, half = d["fold"], d["half"]
    imp = np.zeros((N_BIP, 6)); cnt = np.zeros(6)
    for f in sorted(set(fold[fold >= 0])):
        model = build_model(EXPERIMENTS[name]).to(DEV)
        model.load_state_dict(torch.load(OUT / "models" / f"{name}_f{f}.pt", map_location=DEV)); model.eval()
        te = np.where((fold == f) & (half == 1))[0][: max_n // CFG["n_folds"]]
        lab = Y[te].argmax(1)
        base = softmax_np(predict_logits(model, X, Y, te))[np.arange(len(te)), lab]
        for c in range(N_BIP):
            Xo = np.array(X[te], dtype=np.float16); Xo[:, c] = 0
            po = softmax_np(predict_logits(model, Xo, Y[te], np.arange(len(te))))[np.arange(len(te)), lab]
            for k in range(6):
                imp[c, k] += (base - po)[lab == k].sum()
        cnt += np.bincount(lab, minlength=6)
    imp = pd.DataFrame(imp / np.maximum(cnt, 1), index=BIP_NAMES, columns=CLASSES)
    imp.to_csv(OUT / "results" / f"fig4_occlusion_{name}.csv")
    return imp


if __name__ == "__main__" and Path(CFG["data_dir"]).exists():
    t2, t4 = build_tables(meta, Y)
    print(t2.to_string()); print(t4.to_string())
    if HAS_TORCH and (OUT / "results" / "oof_full18_kl.npz").exists():
        print(channel_occlusion("full18_kl", meta, Y).round(4))

# %% [markdown]
# ## 8. Figures for the manuscript

# %%
def make_figures():
    import matplotlib.pyplot as plt
    t2 = pd.read_csv(OUT / "results" / "table2_point_metrics.csv")
    mean = lambda s: float(str(s).split(" ± ")[0])
    # Fig. 2: performance vs number of channels
    k_map = {"frontal2_kl": 2, "frontotemporal4_kl": 4, "temporal8_kl": 8, "parasagittal8_kl": 8, "full18_kl": 18,
             "learned_k2_kl": 2, "learned_k4_kl": 4, "learned_k6_kl": 6, "learned_k8_kl": 8, "learned_k10_kl": 10}
    groups = {"learned subset": [n for n in k_map if n.startswith("learned")] + ["full18_kl"],
              "fixed montage": ["frontal2_kl", "frontotemporal4_kl", "temporal8_kl", "parasagittal8_kl", "full18_kl"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for (kind, members), mk in zip(groups.items(), ["o-", "s"]):
        sub = t2[t2.Experiment.isin(members)]
        sub = sub.assign(k=sub.Experiment.map(k_map)).sort_values("k")
        ax[0].plot(sub.k, sub.MacroAUROC.map(mean), mk, label=kind)
        ax[1].plot(sub.k, sub.KL.map(mean), mk, label=kind)
    ax[0].set(xlabel="Bipolar channels", ylabel="Macro-AUROC"); ax[1].set(xlabel="Bipolar channels", ylabel="KL divergence")
    for a in ax: a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "results" / "fig2_channels_curve.png", dpi=300)
    # Fig. 4: occlusion heat-map
    f4 = OUT / "results" / "fig4_occlusion_full18_kl.csv"
    if f4.exists():
        imp = pd.read_csv(f4, index_col=0)
        fig, ax = plt.subplots(figsize=(6, 7))
        im = ax.imshow(imp.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(6), imp.columns); ax.set_yticks(range(len(imp)), imp.index)
        fig.colorbar(im, label="Drop in p(majority class)"); fig.tight_layout()
        fig.savefig(OUT / "results" / "fig4_occlusion.png", dpi=300)


if __name__ == "__main__" and (OUT / "results" / "table2_point_metrics.csv").exists():
    make_figures()

# %% [markdown]
# ## 9. (Optional) Kaggle late submission: hidden-test KL score
# Run in a separate notebook with the same code, Internet off, models attached as a dataset.

# %%
def make_submission(name, model_dir=OUT / "models"):
    test = pd.read_csv(f"{CFG['data_dir']}/test.csv")
    Xt = np.stack([load_eeg_window(e, 0.0, split="test") for e in test.eeg_id])
    probs = []
    for f in range(CFG["n_folds"]):
        pth = Path(model_dir) / f"{name}_f{f}.pt"
        if not pth.exists():
            continue
        model = build_model(EXPERIMENTS[name]).to(DEV)
        model.load_state_dict(torch.load(pth, map_location=DEV))
        probs.append(softmax_np(predict_logits(model, Xt, np.zeros((len(Xt), 6), np.float32), np.arange(len(Xt)))))
    sub = pd.DataFrame(np.mean(probs, 0), columns=VOTES); sub.insert(0, "eeg_id", test.eeg_id.values)
    sub.to_csv("submission.csv", index=False)
    return sub
