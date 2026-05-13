import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify_temporal_series(ts, model, window_size=23, stride=1, thres=0.5, class_system={}):
    # ======================================================
    # COPY
    # ======================================================
    df = ts.copy()
    df["Index"] = pd.to_datetime(df["Index"])
    df = (df.sort_values("Index").reset_index(drop=True))

    # ======================================================
    # FEATURE COLUMNS
    # ======================================================
    feature_cols = [col for col in df.columns if col != "Index"]

    # ======================================================
    # LABEL MAP
    # ======================================================
    label_map = dict(zip(class_system["index"], class_system["class_name"]))
    n_classes = len(class_system)

    # ======================================================
    # STORAGE
    # ======================================================
    preds_per_timestamp = [[] for _ in range(len(df))]
    probs_per_timestamp = [[] for _ in range(len(df))]

    # ======================================================
    # CREATE SLIDING WINDOWS
    # ======================================================
    X = []
    window_positions = []
    for start in range(0, len(df) - window_size + 1, stride):

        end = start + window_size
        window = df.iloc[start:end]

        # --------------------------------------------------
        # FLATTEN FEATURES
        # --------------------------------------------------
        features = (window[feature_cols].values.flatten())
        X.append(features)
        window_positions.append(list(range(start, end)))

    # ======================================================
    # NUMPY
    # ======================================================
    X = np.array(X)

    # ======================================================
    # NORMALIZATION
    # ======================================================
    if hasattr(model, "scaler_mean"):
        X = (X - model.scaler_mean) / model.scaler_std

    # ======================================================
    # PREDICT
    # ======================================================
    y_probs, y_preds = model.predict(X, threshold=thres)

    # ======================================================
    # ASSOCIATE WINDOW CLASS TO ALL TIMESTAMPS
    # ======================================================
    for w_idx, positions in enumerate(window_positions):
        pred_class = int(y_preds[w_idx])
        pred_prob = y_probs[w_idx]
        for pos in positions:
            preds_per_timestamp[pos].append(pred_class)
            probs_per_timestamp[pos].append(pred_prob)

    # ======================================================
    # FINAL PREDICTIONS
    # Uses LAST associated window
    # ======================================================
    final_preds = []
    final_probs = []
    for pred_list, prob_list in zip(preds_per_timestamp, probs_per_timestamp):
        # --------------------------------------------------
        # NO PREDICTION
        # --------------------------------------------------
        if len(pred_list) == 0:
            final_preds.append(np.nan)
            final_probs.append(np.full(n_classes, np.nan))
        # --------------------------------------------------
        # USE LAST WINDOW PREDICTION
        # --------------------------------------------------
        else:
            final_preds.append(pred_list[-1])
            final_probs.append(prob_list[-1])

    # ======================================================
    # LABELS
    # ======================================================
    final_labels = []
    for pred in final_preds:
        if pd.isna(pred):
            final_labels.append(None)
        else:
            final_labels.append(label_map.get(int(pred), None))

    # ======================================================
    # OUTPUT
    # ======================================================
    result = df.copy()
    result["pred"] = final_preds
    result["label"] = final_labels

    # probabilities as matrix
    final_probs = np.array(final_probs)
    return result, final_probs

def plot_temporal(time_series, start_date=None, end_date=None):
    # ==========================================
    # COPY
    # ==========================================
    df = time_series.copy()
    df["Index"] = pd.to_datetime(df["Index"])

    # ==========================================
    # OPTIONAL DATE FILTER
    # ==========================================
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["Index"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["Index"] <= end_date]

    # ==========================================
    # FIGURE
    # ==========================================
    fig, ax = plt.subplots(figsize=(18, 6))

    # ==========================================
    # TIME SERIES
    # ==========================================
    ignore_columns = ["Index", "pred", "label"]
    for column in df.columns:
        if column not in ignore_columns:
            ax.plot(
                df["Index"],
                df[column],
                label=column,
                linewidth=2
            )

    # ==========================================
    # STYLE
    # ==========================================
    ax.set_title("Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_temporal_classification(time_series, class_system, title="", start_date=None, end_date=None):
    # ==========================================
    # COPY
    # ==========================================
    df = time_series.copy()
    df["Index"] = pd.to_datetime(df["Index"])

    # ==========================================
    # OPTIONAL DATE FILTER
    # ==========================================
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df["Index"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["Index"] <= end_date]

    # ==========================================
    # COLOR MAP
    # ==========================================
    color_map = dict(zip(class_system["index"], class_system["color"]))
    label_map = dict(zip(class_system["index"], class_system["class_name"]))

    # ==========================================
    # FIGURE
    # ==========================================
    fig, ax = plt.subplots(figsize=(18, 6))

    # ==========================================
    # BACKGROUND COLORS (MERGED SEGMENTS)
    # ==========================================

    valid_df = df.dropna(
        subset=["pred"]
    ).copy()

    valid_df["pred"] = (
        valid_df["pred"]
        .astype(int)
    )

    # inicia primeiro segmento
    start_idx = 0

    for i in range(1, len(valid_df)):

        current_class = valid_df.iloc[i]["pred"]
        previous_class = valid_df.iloc[i - 1]["pred"]

        # mudança de classe
        if current_class != previous_class:

            start_date = valid_df.iloc[start_idx]["Index"]
            end_date = valid_df.iloc[i]["Index"]

            class_id = previous_class

            ax.axvspan(
                start_date,
                end_date,
                color=color_map.get(class_id, "#FFFFFF"),
                alpha=0.25
            )

            start_idx = i

    # último segmento
    last_class = valid_df.iloc[start_idx]["pred"]

    ax.axvspan(
        valid_df.iloc[start_idx]["Index"],
        valid_df.iloc[-1]["Index"],
        color=color_map[last_class],
        alpha=0.25
    )

    # ==========================================
    # TIME SERIES
    # ==========================================
    for column in df.columns:
        if column not in ["Index", "pred", "label"]:
            ax.plot(df["Index"], df[column], label=column, linewidth=2)

    # ==========================================
    # CLASS LEGEND
    # ==========================================
    unique_classes = sorted(valid_df["pred"].unique())
    class_patches = []
    for cls in unique_classes:
        patch = mpatches.Patch(color=color_map[cls], label=label_map[cls], alpha=0.4)
        class_patches.append(patch)

    # ==========================================
    # LEGENDS
    # ==========================================
    signal_legend = ax.legend(loc="upper left")
    ax.add_artist(signal_legend)
    ax.legend(
        handles=class_patches,
        title="Predicted Classes",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    # ==========================================
    # TITLE
    # ==========================================
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def plot_real_trajectory(time_series, trajectory, class_system, collection="mapbiomas-v10", start_date=None, end_date=None):
    # ======================================================
    # PREPARE TIME SERIES
    # ======================================================
    ts = time_series.copy()
    ts["Index"] = pd.to_datetime(ts["Index"])

    # ======================================================
    # OPTIONAL DATE FILTER
    # ======================================================
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        ts = ts[ts["Index"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        ts = ts[ts["Index"] <= end_date]

    # ======================================================
    # PREPARE REAL TRAJECTORY
    # ======================================================
    traj = trajectory.copy()
    traj = traj[(traj["point_id"] == 1) & (traj["collection"] == collection)].copy()
    traj["date"] = pd.to_datetime(traj["date"].astype(str) + "-01-01")

    # ======================================================
    # APPLY SAME DATE FILTER TO TRAJECTORY
    # ======================================================
    if start_date is not None:
        traj = traj[traj["date"] >= start_date]
    if end_date is not None:
        traj = traj[traj["date"] <= end_date]

    # ======================================================
    # COLOR DICTIONARY
    # ======================================================
    color_dict = dict(zip(class_system["class_name"], class_system["color"]))

    # ======================================================
    # PLOT
    # ======================================================
    fig, ax = plt.subplots(figsize=(18, 6))

    # ------------------------------------------------------
    # Background colored sections
    # ------------------------------------------------------
    for i in range(len(traj) - 1):
        start = traj.iloc[i]["date"]
        end = traj.iloc[i + 1]["date"]
        cls = traj.iloc[i]["class"]
        color = color_dict.get(cls, "#FFFFFF")
        ax.axvspan(start, end, color=color, alpha=0.25)

    # ------------------------------------------------------
    # Last class interval
    # ------------------------------------------------------
    if len(traj) > 0:
        last_date = traj.iloc[-1]["date"]
        final_date = (ts["Index"].max() if end_date is None else end_date)
        ax.axvspan(last_date, final_date, color=color_dict.get(traj.iloc[-1]["class"], "#FFFFFF"), alpha=0.25)

    # ======================================================
    # TIME SERIES
    # ======================================================
    for column in time_series.columns:
        if column not in ["Index", "pred", "label"]:
            ax.plot(ts["Index"], ts[column], label=column, linewidth=2)

    # ======================================================
    # LEGEND
    # ======================================================
    class_patches = []
    used_classes = traj["class"].unique()
    for cls in used_classes:
        patch = mpatches.Patch(color=color_dict.get(cls, "#FFFFFF"), label=cls, alpha=0.4)
        class_patches.append(patch)
    signal_legend = ax.legend(loc="upper left")
    ax.add_artist(signal_legend)
    ax.legend(
        handles=class_patches,
        title="Real Classes",
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    # ======================================================
    # STYLE
    # ======================================================
    ax.set_title(f"Real Temporal Trajectory Using {collection}", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Scaled values")
    ax.grid(False)
    plt.tight_layout()
    plt.show()
