 
#%%
print("Executing Code...")
#Import libraries and models
import os, glob, time, zipfile
from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import time
from scipy.signal import welch
import soundfile as sf
import librosa
import shutil
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier as KNeighboursClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Audio, display
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, average_precision_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll.base import scope
from matplotlib.patches import Patch


# change to match absolute file adresses for your pc, r is there to read as string literal
zip_path = r"C:\Users\Ryank\Downloads\Lab 1 Heart Sound Classification.zip"
extract_dir = r"C:\Users\Ryank\OneDrive\Desktop\.Stuff Needed for backup 28th Dec 2023\.Year 4 Engineering\ENGR415 Machine Learning in Engineering\Lab 1 Coursework"
# Create the folder if it does not exist
os.makedirs(extract_dir, exist_ok=True)
# Open the zip file and extract all contents to extract_dir
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir) 
# Print: extraction done, show top-level items
print("Unzipped to:", extract_dir)
print("Top-level items:", os.listdir(extract_dir)[:20]) # check what was extracted so you don’t use a wrong path later.
 
 
# define the file data path
data_root = os.path.join(extract_dir, "Lab 1 Heart Sound Classification", "Data")
 
# Recursively find all files named REFERENCE.csv under data_root
ref_files = glob.glob(os.path.join(data_root, "**", "REFERENCE.csv"), recursive=True)
print("Found REFERENCE.csv:", len(ref_files))
 
AUDIO_EXTS = [".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"]
 
# Declare rows as empty array
rows = []
 
# Process each REFERENCE.csv file one by one
# rf is the path to one REFERENCE.csv
for rf in ref_files:
    folder = os.path.dirname(rf)
    df = pd.read_csv(rf, header=None)
 
    for _, r in df.iterrows():
        name = str(r.iloc[0]).strip()
        label = str(r.iloc[1]).strip()
 
        # 1) Find wav
        audio_path = None
        for ext in AUDIO_EXTS:
            candidate = os.path.join(folder, name + ext)
            if os.path.exists(candidate):
                audio_path = candidate
                break
 
        # 2) Find audios with other extensions
        if audio_path is None:
            candidates = glob.glob(os.path.join(folder, name + ".*"))
            candidates = [c for c in candidates if os.path.splitext(c)[1].lower() in AUDIO_EXTS]
            if len(candidates) > 0:
                audio_path = candidates[0]
 
        if audio_path is not None:
            rows.append({"filepath": audio_path, "label": label})
 
# Convert rows into a table (DataFrame)
meta = pd.DataFrame(rows)
print(meta.head())
print(meta["label"].value_counts())
print("Extension:", meta["filepath"].head().apply(lambda p: os.path.splitext(p)[1]).tolist())
 
 
# Randomly pick 1 "Abnormal" and 1 "Normal "sample from meta
ABNORMAL_LABEL = "1"
NORMAL_LABEL = "-1"
 
abn_row = meta[meta["label"].astype(str).str.strip() == ABNORMAL_LABEL].sample(1, random_state=0).iloc[0]
nor_row = meta[meta["label"].astype(str).str.strip() == NORMAL_LABEL].sample(1, random_state=0).iloc[0]
 
# Read abnormal audio: get signal y and sampling rate sr
abn_path = abn_row["filepath"]
nor_path = nor_row["filepath"]
 
print("Abnormal path:", abn_path)
abn_y, abn_sr = sf.read(abn_path)
display(Audio(abn_y, rate=abn_sr))
 
 
print("Normal path", nor_path)
nor_y, nor_sr = sf.read(nor_path)
display(Audio(nor_y, rate=nor_sr))
 
# Helper function to plot waveform and spectrogram side by side, with consistent styling and saving 
def plot_waveform_and_spectogram(y, sr, title, fmax=1000, show_spectrogram=True, use_log_freq=True, start_sec=0.0, duration_sec=10.0, spec_vmin_db=-80, spec_vmax_db=0, save_dir="."):

    # Check to ensure y is 1D and convert to float
    y = np.asarray(y)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(float)

    # Normalisation
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak

    # Cropping
    start_i = int(max(0, start_sec * sr))
    end_i = int(min(len(y), (start_sec + duration_sec) * sr))
    y_seg = y[start_i:end_i]
    target_len = int(duration_sec * sr)
    if len(y_seg) < target_len:
        y_seg = np.pad(y_seg, (0, target_len - len(y_seg)))
    t = np.arange(len(y_seg)) / sr

    # Plotting
    if show_spectrogram:
        fig = plt.figure(figsize=(10, 6))
        # Define gridspec with 2 rows and 2 columns
        gs = gridspec.GridSpec(2, 2,
            width_ratios=[1.0, 0.035], height_ratios=[1.0, 0.9], wspace=0.05, hspace=0.15) 

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1) # Shared axes for aligned x-axis
        cax = fig.add_subplot(gs[1, 1])
        ax1.tick_params(labelbottom=False)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 3.8))
        ax2, cax = None, None

    # Waveform
    ax1.plot(t, y_seg, linewidth=0.9, alpha=0.9)
    ax1.set_ylabel("Normalised audio amplitude")
    ax1.set_title(f"{title}: Waveform")
    ax1.set_xlim(0, duration_sec)

    # Spectrogram
    # librosa documentation: https://librosa.org/doc/latest/index.html, has some examples as well
    if show_spectrogram and ax2 is not None:
        n_fft = 2048
        hop = 512
        fmin = 20
        S = librosa.stft(y_seg, n_fft=n_fft, hop_length=hop, center=True)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)
        fmask = (freqs >= fmin) & (freqs <= fmax)
        S_db = S_db[fmask, :]
        freqs = freqs[fmask]
        img = ax2.pcolormesh(
        times, freqs, S_db,
        shading="auto",
        vmin=spec_vmin_db, vmax=spec_vmax_db,
        cmap="inferno") # I like inferno ok

        ax2.set_xlim(0, duration_sec)
        ax2.set_xticks(np.arange(0, duration_sec + 0.1, 1.0))
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(f"{title} - Spectrogram (dB)")

        if use_log_freq:
            ax2.set_yscale("log")
            ax2.set_ylim(fmin, fmax)
            yticks = [20, 50, 100, 200, 500, 1000] # Kinda logarithmic, I wanted 20Hz (human hearing lower limit), 1000Hz (upper limit), and some in between for reference
            yticks = [v for v in yticks if fmin <= v <= fmax]
            ax2.set_yticks(yticks)
            ax2.set_yticklabels([str(v) for v in yticks])
        else:
            ax2.set_ylim(0, fmax)

        cbar = fig.colorbar(img, cax=cax)
        cbar.set_label("Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{title} Waveform_Spectrogram.svg"), format="svg", dpi=300, transparent=True)
    plt.show()
    plt.close(fig)
    
# Plot waveforms with the help of that helper function
plot_waveform_and_spectogram(abn_y, abn_sr, "Abnormal", duration_sec=10, use_log_freq=True)
plot_waveform_and_spectogram(nor_y, nor_sr, "Normal",  duration_sec=10, use_log_freq=True)
print("Waveforms plotted")

def plot_psd_overlay(
    y_abn, sr_abn, y_norm, sr_norm, fmax=1000, signal_band=(20, 150), low_band=(20, 150), high_band=(150, 1000), noise_band=(800, 1000), save_path=None): # A lot of these were arbitrarily chosen to compute metrics

    # Since "PSD" in the original lab wasn't actually the PSD, I had to write a helper function for the Welch PSD estimation. See https://uk.mathworks.com/help/signal/ref/pwelch.html
    def welch_psd(y, sr):
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:
            y = y[:, 0]

        nperseg = min(2048, len(y))
        noverlap = min(1024, max(0, nperseg // 2))
        f, Pxx = welch(y,fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, scaling="density")

        mask = f <= fmax
        f = f[mask]
        Pxx = Pxx[mask]
        Pxx_db = 10 * np.log10(Pxx + 1e-20)

        return f, Pxx, Pxx_db

    # Define metrics: bandpower, spectral flatness, roll-off frequency, and a pseudo-SFDR (peak relative to median floor in the noise band)
    def bandpower(f, Pxx, f_lo, f_hi):
        mask = (f >= f_lo) & (f <= f_hi)
        if np.sum(mask) < 2:
            return 0.0
        return np.trapz(Pxx[mask], f[mask])

    def spectral_flatness(Pxx):
        gm = np.exp(np.mean(np.log(Pxx + 1e-20)))
        am = np.mean(Pxx + 1e-20)
        return float(gm / am)

    def rolloff_freq(f, Pxx, roll=0.95):
        c = np.cumsum(Pxx)
        c /= (c[-1] + 1e-20)
        idx = np.searchsorted(c, roll)
        return float(f[min(idx, len(f)-1)])

    def pseudo_sfdr(f, Pxx_db):
        peak = float(np.max(Pxx_db))
        noise_mask = (f >= noise_band[0]) & (f <= noise_band[1])
        if np.sum(noise_mask) < 2:
            floor = float(np.median(Pxx_db))
        else:
            floor = float(np.median(Pxx_db[noise_mask]))
        return peak - floor

    # Compute PSDs
    fA, PxxA, PxxA_db = welch_psd(y_abn, sr_abn)
    fN, PxxN, PxxN_db = welch_psd(y_norm, sr_norm)

    # Compute metrics for signals and store in dataframe
    def compute_metrics(f, Pxx, Pxx_db, label):
        P_sig   = bandpower(f, Pxx, signal_band[0], signal_band[1])
        P_noise = bandpower(f, Pxx, noise_band[0], noise_band[1])

        snr_db = 10 * np.log10((P_sig + 1e-20) / (P_noise + 1e-20))

        P_low  = bandpower(f, Pxx, low_band[0], low_band[1])
        P_high = bandpower(f, Pxx, high_band[0], high_band[1])
        ratio_db = 10 * np.log10((P_low + 1e-20) / (P_high + 1e-20))

        flat   = spectral_flatness(Pxx)
        roll95 = rolloff_freq(f, Pxx, roll=0.95)
        sfdr_db = pseudo_sfdr(f, Pxx_db)

        return {
            "Signal": label,
            f"Band power (signal {signal_band[0]}-{signal_band[1]} Hz)": P_sig,
            f"Band power (noise {noise_band[0]}-{noise_band[1]} Hz)": P_noise,
            "Band SNR (proxy) (dB)": snr_db,
            f"Low-band power ({low_band[0]}-{low_band[1]} Hz)": P_low,
            f"High-band power ({high_band[0]}-{high_band[1]} Hz)": P_high,
            "Low/High ratio (dB)": ratio_db,
            "Spectral flatness (0-1)": flat,
            "95% roll-off frequency (Hz)": roll95,
            "Pseudo-SFDR (dB)": sfdr_db
        }

    metrics_abn  = compute_metrics(fA, PxxA, PxxA_db, "Abnormal")
    metrics_norm = compute_metrics(fN, PxxN, PxxN_db, "Normal")
    df_metrics = pd.DataFrame([metrics_abn, metrics_norm])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(signal_band[0], signal_band[1], alpha=0.07, color="blue")
    ax.plot(fA, PxxA_db, linewidth=1.5, label="Abnormal")
    ax.plot(fN, PxxN_db, linewidth=1.5, label="Normal")
    ax.set_xlim(0, fmax)
    ax.set_xticks(np.arange(0, fmax + 1, 100))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB/Hz)")
    ax.set_title("PSD of normal and abnormal heart sounds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    # Print metrics table
    print("\n=== Frequency-Domain Comparison Metrics ===")
    print(df_metrics.to_string(index=False))
    return df_metrics

df_metrics = plot_psd_overlay(y_abn=abn_y, sr_abn=abn_sr, y_norm=nor_y, sr_norm=nor_sr, fmax=1000, signal_band=(20, 150), noise_band=(800, 1000), low_band=(20, 150), high_band=(150, 1000), save_path="PSD_Overlay_Comparison.svg")
#%%
# Feature extraction, training and evaluation
def extract_features_5_seconds(y, sr, win_sec=5.0):
    y = np.asarray(y)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(float)

    win_len = int(win_sec * sr)
    # If audio is shorter than 5 seconds, pad zeros at the end
    if len(y) < win_len:
        y = np.pad(y, (0, win_len - len(y)))
 
    all_parts = []
    for start in range(0, len(y) - win_len + 1, win_len):
        part = y[start:start+win_len]
 
        # Simple amplitude statistics
        mean = np.mean(part)                 # average amplitude
        std = np.std(part)                   # variability
        mx = np.max(part)                    # maximum
        mn = np.min(part)                    # minimum
        rms = np.sqrt(np.mean(part**2))      # overall energy
        zcr = np.mean(librosa.feature.zero_crossing_rate(part)[0])  # Zero-crossing rate: how often the signal changes sign (roughly how "fast" it changes)
 
        # Frequency-structure features：
        centroid = np.mean(librosa.feature.spectral_centroid(y=part, sr=sr)[0])     # spectral centroid
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=part, sr=sr)[0])   # spectral bandwidth
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=part, sr=sr)[0])       # rolloff frequency
        flatness = np.mean(librosa.feature.spectral_flatness(y=part)[0])            # flatness
 
        # A widely-used audio feature set for classification (MFCC)
        mfcc = librosa.feature.mfcc(y=part, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
 
        one_part_features = np.hstack([
            mean, std, mx, mn, rms, zcr,
            centroid, bandwidth, rolloff, flatness,
            mfcc_mean, mfcc_std
        ])
        all_parts.append(one_part_features)
 
    all_parts = np.vstack(all_parts)
    return np.mean(all_parts, axis=0)
 
# Randomly sample 200 recordings (for quick testing)
sample_meta = meta.sample(200, random_state=67).reset_index(drop=True)
 
# Use lists to store features (X) and labels (y)
X_list, y_list = [], []         #feature and label lists
t0 = time.time()                #Counter start
 
for _, row in sample_meta.iterrows():
    y, sr = sf.read(row["filepath"])
    feats = extract_features_5_seconds(y, sr, win_sec=5.0)
    X_list.append(feats)
    y_list.append(row["label"])
 
X = np.vstack(X_list)                             # number_of_samples × number_of_features
y = np.array([str(label) for label in y_list])    # number_of_samples (as strings)
 
 
print("Feature table shape:", X.shape)
print("Time (seconds):", round(time.time() - t0, 2))

# Split data into two parts: 70% for training, 30% for testing
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0, stratify=y
)

# Define metrics (Sensitivity, Specificity, and custom (mean of the two)) based on confusion matrix
NORMAL_LABEL = "-1"
ABNORMAL_LABEL = "1"

def se_sp_custom(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[NORMAL_LABEL, ABNORMAL_LABEL])
    tn, fp, fn, tp = cm.ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    custom = 0.5 * (se + sp)
    return tn, fp, fn, tp, se, sp, custom

def custom_accuracy_se_sp(y_true, y_pred):
    return se_sp_custom(y_true, y_pred)[6]

custom_scorer = make_scorer(custom_accuracy_se_sp)
scoring = {"custom": custom_scorer, "accuracy": "accuracy"}

def evaluate_on_test(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    tn, fp, fn, tp, se, sp, custom = se_sp_custom(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return custom, acc, se, sp, y_pred

# Confusion matrix code
# This is plug and play basically, don't ask me how it works, just use it to plot confusion matrices with the same style for all models. You can change the font sizes and colours if you want, but the layout is fixed to be tight and compact.
def plot_confusion_matrix_detailed(
    y_true, y_pred, model_name, save_path=None,
    cmap="RdYlGn",
    fontsize_cells=13,
    fontsize_boxes=8,
    legend=True
):
    cm = confusion_matrix(y_true, y_pred, labels=[NORMAL_LABEL, ABNORMAL_LABEL])
    tn, fp, fn, tp = cm.ravel()

    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / np.sum(cm) if np.sum(cm) else 0.0
    f1  = f1_score(y_true, y_pred, pos_label=ABNORMAL_LABEL, zero_division=0)

    # Colour semantic stuff
    tnr = sp
    fpr = 1 - sp
    fnr = 1 - se
    tpr = se
    rate_matrix = np.array([[ tnr, -fpr ],
                            [ -fnr, tpr ]])
    # Layout stuff

    nrows = 4 + (1 if legend else 0) # 4 rows for CM, Se/Sp, Acc, F1 + optional legend row
    fig_h = 6.6 if legend else 6.1 
    fig = plt.figure(figsize=(10.8, fig_h)) 
    fig.tight_layout(pad=0.3) 
    plt.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10) 

    # Figure and main confusion matrix
    fig_h = 6.4 if legend else 6.0
    fig = plt.figure(figsize=(9.2, fig_h)) 

    # Main confusion matrix axis (large, on the left)
    ax_cm = fig.add_axes([0.08, 0.33, 0.62, 0.58])  # [left, bottom, width, height]

    # Get CM axis bbox to anchor everything else relative to it
    bbox = ax_cm.get_position()
    x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
    x1, y1 = bbox.x1, bbox.y1

    # Anchor function for stat boxes (Se/Sp, Acc, F1, PPV/NPV)
    def add_stat_ax(left, bottom, width, height):
        ax = fig.add_axes([left, bottom, width, height])
        ax.axis("off")
        return ax

    # Hug the confusion matrix
    gap = 0.003
    rhs_w = 0.04
    box_h = 0.13
    rhs_left = x1 + gap

    # Centres of top and bottom halves of the confusion matrix
    y_top_center = y0 + 0.75*h
    y_bot_center = y0 + 0.25*h

    ax_se = add_stat_ax(rhs_left, y_top_center - box_h/2, rhs_w, box_h)
    ax_sp = add_stat_ax(rhs_left, y_bot_center - box_h/2, rhs_w, box_h)

    bottom_gap = 0.14
    ppv_h = 0.09
    col_gap = 0.01
    ax_acc = add_stat_ax(rhs_left, y0 - bottom_gap, rhs_w, ppv_h)

    acc_w = rhs_w * 0.95
    f1_w  = rhs_w * 0.95
    mini_gap = 0.05

    # F1
    ax_f1 = add_stat_ax(
        rhs_left + acc_w + mini_gap,
        y0 - bottom_gap,
        f1_w,
        ppv_h
    )

    ax_ppv = add_stat_ax(x0, y0 - bottom_gap, (w/2) - col_gap/2, ppv_h)
    ax_npv = add_stat_ax(x0 + (w/2) + col_gap/2, y0 - bottom_gap, (w/2) - col_gap/2, ppv_h)

    # Legend
    ax_leg = None
    if legend:
        leg_h = 0.10
        ax_leg = add_stat_ax(x0, y0 - bottom_gap - leg_h - 0.02, w + rhs_w + gap, leg_h)

    # Confusion matrix with coloured cells
    im = ax_cm.imshow(rate_matrix, cmap=cmap, vmin=-1, vmax=1)
    ax_cm.set_aspect("equal")

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Normal (-1)", "Pred: Abnormal (1)"])
    ax_cm.set_yticklabels(["True: Normal (-1)", "True: Abnormal (1)"])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title(f"Confusion Matrix – {model_name}")

    # Cell labels
    cell_labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"]
    ])
    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j, i, cell_labels[i, j],
                ha="center", va="center",
                color="black",
                fontsize=fontsize_cells,
                fontweight="bold"
            )

    def stat_box(ax, title, formula, value, align="center"):
        ax.axis("off")

        if align == "left":
            x, ha = 0.02, "left"    
        else:
            x, ha = 0.5, "center" 

        ax.text(
            x, 0.5,
            f"{title}\n{formula}\n= {value:.3f}",
            ha=ha, va="center",
            fontsize=fontsize_boxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95)
        )

    stat_box(ax_se,  "Sensitivity", "TP / (TP + FN)", se)
    stat_box(ax_sp,  "Specificity", "TN / (TN + FP)", sp)
    stat_box(ax_acc, "Accuracy", "(TP + TN) / All", acc)
    stat_box(ax_f1,  "F1 Score", "2PR / (P + R)", f1)

    stat_box(ax_ppv, "Precision", "TP/(TP+FP)", ppv)
    stat_box(ax_npv, "NPV", "TN/(TN+FN)", npv)

    if legend:
        ax_leg.axis("off")
        cmap_obj = plt.cm.get_cmap(cmap)
        handles = [
            Patch(facecolor=cmap_obj(0.95), edgecolor="black", label="Good (TN/TP rates high)"),
            Patch(facecolor=cmap_obj(0.55), edgecolor="black", label="Neutral (near 0)"),
            Patch(facecolor=cmap_obj(0.10), edgecolor="black", label="Bad (FP/FN rates high)")
        ]
        ax_leg.legend(handles=handles, loc="center", ncol=3, frameon=True, title="Colour meaning")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

# Baseline model comparison (no HPO, default params)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "KNN": KNeighboursClassifier(),
    "SVM": SVC(kernel="rbf"),
    "SGDC": SGDClassifier(random_state=0),
    "HGB": HistGradientBoostingClassifier(random_state=0),
    "GNB": GaussianNB(),
    "MLP": MLPClassifier(activation="logistic", random_state=0, max_iter=2000, early_stopping=True),
} # Optionally you can choose to add early_stopping = True for My Little Pony Classifier

# Store results here
results = []
trained_models = {}

# Iterate through each model, train, predict, evaluate, and store results
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

    # Train, time
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict, time
    t0 = time.time()
    y_pred = pipe.predict(X_test)
    predict_time = time.time() - t0

    # Recall custom accuracy score (and accuracy for reference)
    custom_acc = custom_accuracy_se_sp(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Add to df
    results.append([name, custom_acc, acc, train_time, predict_time])
    trained_models[name] = pipe

# This is the df, sort by custom accuracy score (descending) and reset index for better display
df_res = pd.DataFrame(results, columns=["Method", "CustomAcc(Se/Sp)", "Accuracy", "Train time (s)", "Predict time (s)"])
df_res = df_res.sort_values("CustomAcc(Se/Sp)", ascending=False).reset_index(drop=True)
print("\nBaseline model comparison")
print(df_res)
 
# Pick the best method by score
best_name = df_res.iloc[0]["Method"]
best_model = trained_models[best_name]
 
y_pred = best_model.predict(X_test)

# Use big ass helper function to plot confusion matrix
plot_confusion_matrix_detailed(
    y_test,
    y_pred,
    model_name=f"Baseline best: {best_name}",
    save_path="1.Baseline_Best_Model_Confusion_Matrix.svg",
    cmap="RdYlGn",
)

#%%

# Hyperparmater Optimisation bit here
# Setup pipelines here

pipelines = {
    "GNB": Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB())]),
    "DTC": Pipeline([("scaler", StandardScaler()), ("classifier", DecisionTreeClassifier(random_state=0))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf"))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighboursClassifier())]),
    "HGB": Pipeline([("scaler", StandardScaler()), ("classifier", HistGradientBoostingClassifier(random_state=0))]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier(activation="logistic", random_state=0, max_iter=2000, early_stopping=True))]),
} # Here early stopping is turned on so it doesn't take 100 million years

# HPO search spaces for Random Search (I will call this RS from now on) and Tree-based Parzen Estimators (TPE)
param_dists = {
    "GNB": {"classifier__var_smoothing": np.logspace(-12, -6, 50)},
    "DTC": {"classifier__max_depth": np.arange(10, 101),
            "classifier__max_features": np.linspace(0.1, 1.0, 10),
            "classifier__min_samples_split": np.arange(2, 21),
            "classifier__min_samples_leaf": np.arange(1, 21)},
    "SVM": {"classifier__C": np.logspace(-3, 3, 50),
            "classifier__gamma": np.logspace(-4, 1, 50)},
    "KNN": {"classifier__n_neighbors": np.arange(1, 51),
            "classifier__p": np.arange(1, 6)},
    "HGB": {"classifier__max_iter": np.arange(50, 301),
            "classifier__learning_rate": np.linspace(0.01, 0.5, 50),
            "classifier__max_depth": np.arange(1, 31)},
    "MLP": {"classifier__hidden_layer_sizes": np.arange(20, 501, 20),
            "classifier__alpha": np.logspace(-6, -2, 50),
            "classifier__learning_rate_init": np.logspace(-4, -1, 50),
            "classifier__max_iter": np.arange(200, 2001, 200)},
} # All these spaces were mentioned and in appendix B, I think

# Define the manner of which to divide the spaces (int, uniform, loguniform etc)
spaces = {
    "GNB": {"var_smoothing": hp.loguniform("gnb_var_smoothing", np.log(1e-12), np.log(1e-6))},
    "DTC": {"max_depth": scope.int(hp.quniform("dtc_max_depth", 10, 100, 1)),
            "max_features": hp.uniform("dtc_max_features", 0.1, 1.0),
            "min_samples_split": scope.int(hp.quniform("dtc_min_samples_split", 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform("dtc_min_samples_leaf", 1, 20, 1))},
    "SVM": {"C": hp.loguniform("svm_C", np.log(1e-3), np.log(1e3)),
            "gamma": hp.loguniform("svm_gamma", np.log(1e-4), np.log(1e1))},
    "KNN": {"n_neighbors": scope.int(hp.quniform("knn_n_neighbors", 1, 50, 1)),
            "p": scope.int(hp.quniform("knn_p", 1, 5, 1))},
    "HGB": {"max_iter": scope.int(hp.quniform("hgb_max_iter", 50, 300, 1)),
            "learning_rate": hp.uniform("hgb_learning_rate", 0.01, 0.5),
            "max_depth": scope.int(hp.quniform("hgb_max_depth", 1, 30, 1))},
    "MLP": {"hidden_layer_sizes": scope.int(hp.quniform("mlp_hidden_layer_sizes", 20, 500, 10)),
            "alpha": hp.loguniform("mlp_alpha", np.log(1e-6), np.log(1e-2)),
            "learning_rate_init": hp.loguniform("mlp_lr", np.log(1e-4), np.log(1e-1)),
            "max_iter": scope.int(hp.quniform("mlp_max_iter", 200, 2000, 100))},
}

# Self explanatory
def make_pipeline_from_params(model_name, params):
    if model_name == "GNB":
        clf = GaussianNB(var_smoothing=params["var_smoothing"])
    elif model_name == "DTC":
        clf = DecisionTreeClassifier(random_state=0, **params)
    elif model_name == "SVM":
        clf = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"])
    elif model_name == "KNN":
        clf = KNeighboursClassifier(n_neighbors=params["n_neighbors"], p=params["p"])
    elif model_name == "HGB":
        clf = HistGradientBoostingClassifier(random_state=0, **params)
    elif model_name == "MLP":
        clf = MLPClassifier(activation="logistic", random_state=0, hidden_layer_sizes=params["hidden_layer_sizes"], alpha=params["alpha"], learning_rate_init=params["learning_rate_init"], max_iter=params["max_iter"])
    else:
        raise ValueError(model_name)

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])

# Define hyperopt objective function, minimising the negative of the CV custom Sp/Se score
def hyperopt_objective(model_name):
    def _obj(params):
        pipe = make_pipeline_from_params(model_name, params)
        cv_score = cross_val_score(pipe, X_train, y_train, cv=3, scoring=custom_scorer).mean()
        return {"loss": -cv_score, "status": STATUS_OK} # Yeah so make sure that this is negative. You will see the "loss" value in the trials object
    return _obj

# Run HPO for RS and TPE.
RANDOM_N_ITER = 25 # Number of RS iterations
TPE_MAX_EVALS = 25 # NUmber of TPE iterations
# For this report it's probably best to keep these the same

# And then store the results here
random_best_models, tpe_best_models = {}, {}
random_best_params, tpe_best_params = {}, {}
rows = []

# Random search function
for model_name in pipelines.keys():
    rs = RandomizedSearchCV(
        estimator=pipelines[model_name],
        param_distributions=param_dists[model_name],
        n_iter=RANDOM_N_ITER,
        scoring=scoring,
        refit="custom",
        cv=3,
        random_state=0,
        n_jobs=-1
    )
    # Fit time
    t0 = time.time()
    rs.fit(X_train, y_train)
    rs_time = time.time() - t0

    random_best_models[model_name] = rs.best_estimator_
    random_best_params[model_name] = rs.best_params_

    test_custom, test_acc, test_se, test_sp, _ = evaluate_on_test(rs.best_estimator_, X_test, y_test)
     # Add ts to the df
    rows.append({
        "Model": model_name,
        "Method": "RandomSearchCV",
        "Time (s)": rs_time,
        "Best CV Custom": rs.best_score_,
        "Test Custom": test_custom,
        "Test Accuracy": test_acc,
        "Test Sensitivity": test_se,
        "Test Specificity": test_sp,
    })

    # TPE
    # Same as RS basically
    trials = Trials()
    t0 = time.time()
    best = fmin(
        fn=hyperopt_objective(model_name),
        space=spaces[model_name],
        algo=tpe.suggest,
        max_evals=TPE_MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(0),
        verbose=False
    )
    tpe_time = time.time() - t0

    best_params = space_eval(spaces[model_name], best)
    tpe_model = make_pipeline_from_params(model_name, best_params)
    tpe_model.fit(X_train, y_train)

    tpe_best_models[model_name] = tpe_model
    tpe_best_params[model_name] = best_params

    best_cv_custom = -min(t["result"]["loss"] for t in trials.trials)
    test_custom, test_acc, test_se, test_sp, _ = evaluate_on_test(tpe_model, X_test, y_test)

    rows.append({
        "Model": model_name,
        "Method": "TPE (hyperopt)",
        "Time (s)": tpe_time,
        "Best CV Custom": best_cv_custom,
        "Test Custom": test_custom,
        "Test Accuracy": test_acc,
        "Test Sensitivity": test_se,
        "Test Specificity": test_sp,
    })

df_hpo = pd.DataFrame(rows).sort_values(["Test Custom", "Test Accuracy"], ascending=False).reset_index(drop=True)
print("\n Hyperparameter Optimisation results")
print(df_hpo)

# Display the best models, cause we are winners
best_row = df_hpo.iloc[0]
best_model_name = best_row["Model"]
best_method = best_row["Method"]

if best_method == "RandomSearchCV":
    best_estimator = random_best_models[best_model_name]
    best_params = random_best_params[best_model_name]
else:
    best_estimator = tpe_best_models[best_model_name]
    best_params = tpe_best_params[best_model_name]

print("\nBest model overall:")
print(f"Model:         {best_model_name}")
print(f"Method:        {best_method}")
print(f"Best CV Custom: {best_row['Best CV Custom']:.6f}")
print(f"Test Custom:    {best_row['Test Custom']:.6f}")
print(f"Test Accuracy:  {best_row['Test Accuracy']:.6f}")
print(f"Test Sensitivity:        {best_row['Test Sensitivity']:.6f}")
print(f"Test Specificity:        {best_row['Test Specificity']:.6f}")
print("\nBest Params (winner only):")
print(best_params)

_, _, _, _, y_pred_best = evaluate_on_test(best_estimator, X_test, y_test)

# Use big ass helper function for best model too
plot_confusion_matrix_detailed(
    y_test, y_pred_best,
    model_name=f"{best_model_name} ({best_method})",
    save_path="2.Optimised_Best_Model_Confusion_Matrix.svg",
    cmap="RdYlGn",
)
 
#%%
# Appendix plots (you can basically ignore this unless you desperately need the images)

# Calculate score for PR curve
def get_score_for_pr(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return s
    return None

# Permutation importance
def plot_permutation_importance(
    estimator, X_test, y_test,
    feature_names=None,
    n_repeats=30,
    scoring=None,
    top_k=5,
    random_state=0,
    save_path="Appendix_Permutation_Importance.svg"
):

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    if scoring is None:
        scoring = custom_scorer

    r = permutation_importance(
        estimator, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    imp_mean = r.importances_mean
    imp_std  = r.importances_std

    idx = np.argsort(imp_mean)[::-1]
    idx = idx[:top_k]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        imp_mean[idx][::-1]
    )
    ax.set_xlabel("Permutation importance (mean score decrease)")
    ax.set_title(f"Permutation Feature Importance (top {top_k})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    df = pd.DataFrame({
        "feature": [feature_names[i] for i in range(len(feature_names))],
        "importance_mean": imp_mean,
        "importance_std": imp_std
    }).sort_values("importance_mean", ascending=False)

    print("\n=== Permutation Feature Importance (top rows) ===")
    print(df.head(20).to_string(index=False))

    return df

# Tree intrinsic importance (only for tree-based models, no retraining needed, but can be misleading and biased)
def plot_tree_intrinsic_importance(
    estimator,
    feature_names=None,
    top_k=5,
    save_path="Appendix_Tree_Intrinsic_Importance.svg"
):
    clf = estimator
    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        clf = estimator.named_steps["classifier"]

    if not hasattr(clf, "feature_importances_"):
        print("\n(Tree intrinsic importance skipped: model has no feature_importances_.)")
        return None

    importances = clf.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
    ax.set_xlabel("Impurity-based importance")
    ax.set_title(f"Tree Intrinsic Feature Importance (top {top_k})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n=== Tree intrinsic importance (top rows) ===")
    print(df.head(20).to_string(index=False))

    return df

# Distance correlation matrix, with dummy
def _distance_corr(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    n = x.shape[0]
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2 = (A * B).sum() / (n * n)
    dvarx = (A * A).sum() / (n * n)
    dvary = (B * B).sum() / (n * n)

    if dvarx <= 1e-20 or dvary <= 1e-20:
        return 0.0
    return np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx) * np.sqrt(dvary))


def plot_distance_correlation_matrix_with_dummy(
    X, y,
    feature_names=None,
    add_dummy=True,
    random_state=0,
    save_path="Appendix_DistanceCorrelation_withDummy.svg"
):
    rng = np.random.default_rng(random_state)
    X_use = np.array(X, dtype=float)
    if add_dummy:
        dummy = rng.normal(size=(X_use.shape[0], 1))
        X_use = np.hstack([X_use, dummy])

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    if add_dummy:
        feature_names = list(feature_names) + ["DummyNoise"]
    y_num = np.array([1 if str(v) == ABNORMAL_LABEL else 0 for v in y], dtype=float)
    Z = np.hstack([X_use, y_num.reshape(-1, 1)])
    names = feature_names + ["Target"]
    m = Z.shape[1]
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            if i == j:
                D[i, j] = 1.0
            elif j < i:
                D[i, j] = D[j, i]
            else:
                D[i, j] = _distance_corr(Z[:, i], Z[:, j])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(D, vmin=0, vmax=1)

    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    ax.set_title("Distance Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distance correlation")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    target_idx = names.index("Target")
    dummy_idx = names.index("DummyNoise") if add_dummy else None
    if dummy_idx is not None:
        print("\n=== Distance correlation sanity check ===")
        print(f"dCor(DummyNoise, Target) = {D[dummy_idx, target_idx]:.4f}")
        # show top 10 features w.r.t. target
        feat_target = [(names[i], D[i, target_idx]) for i in range(len(feature_names))]
        feat_target = sorted(feat_target, key=lambda t: t[1], reverse=True)
        print("Top features by dCor(feature, Target):")
        for n, v in feat_target[:10]:
            print(f"  {n:>20s} : {v:.4f}")

    return D, names

# PR curve
def plot_precision_recall(
    estimator, X_test, y_test,
    save_path="Appendix_PrecisionRecall.svg"
):
    y_true_bin = np.array([1 if str(v) == ABNORMAL_LABEL else 0 for v in y_test], dtype=int)
    scores = get_score_for_pr(estimator, X_test)
    if scores is None:
        print("\n(PR curve skipped: estimator has neither predict_proba nor decision_function.)")
        return None
    precision, recall, thr = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {ap:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"\n=== Average Precision (AP) ===\nAP = {ap:.4f}")
    return ap

# Pareto Front for Se/Sp
def pareto_front(points):
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    is_opt = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_opt[i]:
            continue
        dominates = (P >= P[i]).all(axis=1) & (P > P[i]).any(axis=1)
        if np.any(dominates):
            is_opt[i] = False
    return is_opt

def plot_pareto_se_sp(df_hpo, save_path="Appendix_Pareto_Se_Sp.svg", zoom=True):
    # Required columns (your df has these)
    se = df_hpo["Test Sensitivity"].astype(float).values
    sp = df_hpo["Test Specificity"].astype(float).values
    method = df_hpo["Method"].astype(str).values

    points = np.column_stack([se, sp])
    opt = pareto_front(points)

    is_rs  = (df_hpo["Method"] == "RandomSearchCV").values
    is_tpe = (df_hpo["Method"] == "TPE (hyperopt)").values

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(se[is_rs],  sp[is_rs],  alpha=0.85, label="Random Search (RS)")
    ax.scatter(se[is_tpe], sp[is_tpe], alpha=0.85, label="TPE (Hyperopt)")
    ax.scatter(
        se[opt], sp[opt],
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=2.8,
        label="Pareto-optimal"
    )
    pareto_idx = np.where(opt)[0]
    pareto_sorted = pareto_idx[np.argsort(se[pareto_idx])]
    ax.plot(
        se[pareto_sorted],
        sp[pareto_sorted],
        linestyle="--",
        linewidth=2.2,
        color="black"
    )
    for i in pareto_sorted:
        label = f"{df_hpo.iloc[i]['Model']} | {df_hpo.iloc[i]['Method']}"
        ax.annotate(label, (se[i], sp[i]), fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Sensitivity (Recall)")
    ax.set_ylabel("Specificity")
    ax.set_title("Pareto Front of Sensitivity and Specificity")
    ax.grid(True, alpha=0.3)

    if zoom:
        ax.set_xlim(max(0, se.min() - 0.05), min(1, se.max() + 0.05))
        ax.set_ylim(max(0, sp.min() - 0.05), min(1, sp.max() + 0.05))
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    pareto_df = df_hpo.loc[opt].copy().sort_values("Test Sensitivity")
    print("\n=== Pareto-optimal points (Se, Sp) ===")
    print(pareto_df[["Model","Method","Best CV Custom","Test Custom","Test Accuracy","Test Sensitivity","Test Specificity"]].to_string(index=False))

    return pareto_df

# Run the appendix plots
feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

print("\n--- Appendix: Feature importance (Permutation) ---")
df_perm = plot_permutation_importance(
    best_estimator, X_test, y_test,
    feature_names=feature_names,
    scoring=custom_scorer,
    top_k=5,
    save_path="Appendix_Permutation_Importance.svg"
)

print("\n--- Appendix: Feature importance (Tree intrinsic if available) ---")
df_tree = plot_tree_intrinsic_importance(
    best_estimator,
    feature_names=feature_names,
    top_k=5,
    save_path="Appendix_Tree_Intrinsic_Importance.svg"
)

print("\n--- Appendix: Distance correlation matrix (+ dummy noise feature) ---")
D, names = plot_distance_correlation_matrix_with_dummy(
    X, y,
    feature_names=feature_names,
    add_dummy=True,
    save_path="Appendix_DistanceCorrelation_withDummy.svg"
)

print("\n--- Appendix: Precision-Recall curve (best model) ---")
ap = plot_precision_recall(
    best_estimator, X_test, y_test,
    save_path="Appendix_PrecisionRecall.svg"
)

print("\n--- Appendix: Pareto front (Sensitivity vs Specificity) ---")
pareto_df = plot_pareto_se_sp(df_hpo, save_path="Appendix_Pareto_Se_Sp.svg", zoom=True)
#%%
#input("Press Enter to exit, unzipped files will be deleted")
#shutil.rmtree(extract_dir, ignore_errors=True)
 
# Copilot's auto generated comment suggestor is a real one (and also the debugger as well)