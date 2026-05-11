import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classify_temporal_series(ts, model, window_size = 23, class_system = {}):
    # ==========================================
    # SORT DATES
    # ==========================================
    df = ts.copy()
    df["Index"] = pd.to_datetime(df["Index"])
    df = df.sort_values("Index")
    df = df.reset_index(drop=True)

    # ==========================================
    # COLOR MAP
    # ==========================================
    label_map = dict(zip(class_system["index"], class_system["class_name"]))

    # ==========================================
    # CREATE WINDOWS
    # ==========================================
    X = []
    pred_dates = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        features = []
        for _, row in window.iterrows():
            all_features = []
            for key in ts.keys():
                if key != "Index":
                    all_features.append(row[key])
            features.extend(all_features)
        X.append(features)
        pred_dates.append(window.iloc[-1]["Index"])
    X = np.array(X)

    # ==========================================
    # NORMALIZATION
    # ==========================================
    if hasattr(model, "scaler_mean"):
        X = (X - model.scaler_mean) / model.scaler_std

    # ==========================================
    # PREDICT
    # ==========================================
    y_probs, y_preds = model.predict(X)

    # ==========================================
    # MAP LABEL NAMES
    # ==========================================
    if label_map is not None:
        labels = [label_map.get(p, None) for p in y_preds]
    else:
        labels = y_preds

    # ==========================================
    # CREATE OUTPUT DF
    # ==========================================
    pred_df = pd.DataFrame({
        "Index": pred_dates,
        "pred": y_preds,
        "label": labels
    })

    # ==========================================
    # MERGE WITH ORIGINAL SERIES
    # ==========================================
    result = df.merge(pred_df, on="Index", how="left")

    return result, y_probs

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
                color=color_map[class_id],
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
        color = color_dict.get(cls, "#DDDDDD")
        ax.axvspan(start, end, color=color, alpha=0.25)

    # ------------------------------------------------------
    # Last class interval
    # ------------------------------------------------------
    if len(traj) > 0:
        last_date = traj.iloc[-1]["date"]
        final_date = (ts["Index"].max() if end_date is None else end_date)
        ax.axvspan(last_date, final_date, color=color_dict.get(traj.iloc[-1]["class"],"#DDDDDD"), alpha=0.25)

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
        patch = mpatches.Patch(color=color_dict.get(cls, "#DDDDDD"), label=cls, alpha=0.4)
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
