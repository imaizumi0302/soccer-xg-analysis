import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import datetime as dt
import ast
from shapely.geometry import Point, Polygon
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

# ==========================================
# 1. Feature Engineering (Class & Functions)
# ==========================================

def filter_penalties(df):
    """Exclude penalty kicks"""
    return df[df["shot_type_name"] != "Penalty"].copy()

def create_goal_flag(df):
    """Create goal flag"""
    df["is_goal"] = (df["shot_outcome_name"] == "Goal").astype(int)
    return df

def handle_missing_values(df):
    """Handle missing values"""
    df = df.copy()
    if "under_pressure" in df.columns:
        df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["x"] = df["x"].fillna(0)
    df["y"] = df["y"].fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def add_dummy_features(df):
    """Perform one-hot encoding in batch"""
    df = df.copy()
    if "shot_body_part_name" in df.columns:
        df = pd.get_dummies(df, columns=["shot_body_part_name"], prefix="body", dtype=int)
    if "shot_type_name" in df.columns:
        df = pd.get_dummies(df, columns=["shot_type_name"], prefix="shoot_type", dtype=int)
    if "play_pattern_name" in df.columns:
        df = pd.get_dummies(df, columns=["play_pattern_name"], prefix="pattern", dtype=int)
    if "shot_technique_name" in df.columns:
        df = pd.get_dummies(df, columns=["shot_technique_name"], prefix="shoot_technic", dtype=int)
    
    return df

class AdvancedFeatureEngineering:
    """Class to generate geometric features, assist features, and freeze-frame features"""

    GOAL_X = 120
    GOAL_CENTER_Y = 40
    LEFT_POST_Y = 36
    RIGHT_POST_Y = 44

    @classmethod
    def preprocess(cls, df):
        df = df.copy()
        df = cls._add_basic_geometry(df)
        df = cls._add_advanced_geometry(df)
        df = cls._add_assist_features(df)
        df = cls._add_freeze_frame_features(df)
        df = cls._clean_data(df)
        return df

    @classmethod
    def _add_basic_geometry(cls, df):
        df["shot_distance"] = np.sqrt((cls.GOAL_X - df["x"])**2 + (cls.GOAL_CENTER_Y - df["y"])**2)
        a_x = cls.GOAL_X - df["x"]
        a_y = cls.LEFT_POST_Y - df["y"]
        b_x = cls.GOAL_X - df["x"]
        b_y = cls.RIGHT_POST_Y - df["y"]
        dot = a_x * b_x + a_y * b_y
        mag_a = np.sqrt(a_x**2 + a_y**2)
        mag_b = np.sqrt(b_x**2 + b_y**2)
        valid_mask = (mag_a * mag_b) > 0
        df["shot_angle"] = 0.0
        df.loc[valid_mask, "shot_angle"] = np.arccos((dot[valid_mask] / (mag_a[valid_mask] * mag_b[valid_mask])).clip(-1, 1))
        df["shot_angle_deg"] = np.degrees(df["shot_angle"])
        return df

    @classmethod
    def _add_advanced_geometry(cls, df):
        df["effective_goal_width"] = 2 * df["shot_distance"] * np.tan(df["shot_angle"] / 2)
        df["angle_from_center"] = np.abs(df["y"] - cls.GOAL_CENTER_Y)
        df["in_penalty_area"] = ((df["x"] >= 102) & (df["y"] >= 18) & (df["y"] <= 62)).astype(int)
        df["in_six_yard_box"] = ((df["x"] >= 114) & (df["y"] >= 30) & (df["y"] <= 50)).astype(int)
        df["distance_angle_interaction"] = df["shot_distance"] * df["shot_angle"]
        df["distance_squared"] = df["shot_distance"] ** 2
        return df

    @classmethod
    def _add_assist_features(cls, df):
        df["has_assist"] = df["assist_x"].notnull().astype(int)
        df["pass_length"] = np.sqrt((df["x"] - df["assist_x"])**2 + (df["y"] - df["assist_y"])**2).fillna(0)
        df["pass_progress_x"] = (df["x"] - df["assist_x"]).fillna(0)
        bool_cols = ["is_cross", "is_cutback", "is_through_ball"]
        for col in bool_cols:
            if col in df.columns: df[col] = df[col].fillna(False).astype(int)
            else: df[col] = 0
        if "pass_height" in df.columns:
            df["is_pass_ground"] = (df["pass_height"] == "Ground Pass").astype(int)
            df["is_pass_high"] = (df["pass_height"] == "High Pass").astype(int)
            df["is_pass_low"] = (df["pass_height"] == "Low Pass").astype(int)

        else:
            df["is_pass_ground"] = 0
            df["is_pass_high"] = 0
            df["is_pass_low"] = 0
            
        return df

    @classmethod
    def _add_freeze_frame_features(cls, df):
        gk_distances = []
        opponents_in_cones = []
        teammates_in_cones = []

        print("🥶 Calculating Freeze Frame features (Universal fixed version)...")

        for i, row in df.iterrows():
            freeze_frame = row.get("shot_freeze_frame")
            shooter_loc = [row["x"], row["y"]]

            # 1. Check validity of the data itself
            # Check whether the data is a list type
            # If it is a list type, accept it and skip further checks
            if isinstance(freeze_frame, list):
                pass
            # If data is NaN or string "nan", fill everything with NaN
            elif pd.isnull(freeze_frame) or (isinstance(freeze_frame, str) and freeze_frame == "nan"):
                gk_distances.append(np.nan); opponents_in_cones.append(np.nan); teammates_in_cones.append(np.nan)
                continue

            # If data is string and not "nan", convert to list type
            elif isinstance(freeze_frame, str):
                try: 
                    freeze_frame = ast.literal_eval(freeze_frame)
                    if not isinstance(freeze_frame, list): raise ValueError
                except:
                    gk_distances.append(np.nan); opponents_in_cones.append(np.nan); teammates_in_cones.append(np.nan)
                    continue

            # For all other cases, fill everything with NaN
            else:
                gk_distances.append(np.nan); opponents_in_cones.append(np.nan); teammates_in_cones.append(np.nan)
                continue
            
            gk_loc = None
            opponent_count = 0
            teammate_count = 0

            # Define triangular area in front of the goal
            triangle_poly = Polygon([
                shooter_loc,
                [cls.GOAL_X, cls.LEFT_POST_Y],
                [cls.GOAL_X, cls.RIGHT_POST_Y]
            ])

            for player in freeze_frame:
                # --- Logic to absorb differences in data structure ---
                
                # Get coordinates
                if "location" in player:
                    player_loc = player["location"] # PL format
                elif "x" in player and "y" in player:
                    player_loc = [player["x"], player["y"]] # WC format
                else:
                    continue

                # Get position name
                if "position" in player and isinstance(player["position"], dict):
                    pos_name = player["position"].get("name", "") # PL format
                elif "position_name" in player:
                    pos_name = player["position_name"] # WC format
                else:
                    pos_name = ""

                is_teammate = player.get("teammate", False)
                # ---------------------------------------

                # Identify GK position
                if (not is_teammate) and (pos_name == "Goalkeeper"):
                    gk_loc = player_loc
                    continue

                # Count number of players inside the cone
                if triangle_poly.contains(Point(player_loc[0], player_loc[1])):
                    if is_teammate: teammate_count += 1
                    else: opponent_count += 1

            # Store features
            if gk_loc: 
                gk_distances.append(np.sqrt((shooter_loc[0]-gk_loc[0])**2 + (shooter_loc[1]-gk_loc[1])**2))
            else: 
                gk_distances.append(np.nan)

            opponents_in_cones.append(opponent_count)
            teammates_in_cones.append(teammate_count)

        df["gk_distance_to_shooter"] = gk_distances
        df["num_opponents_in_shot_cone"] = opponents_in_cones
        df["num_teammates_in_shot_cone"] = teammates_in_cones

        # Fill missing values
        mask_nan_gk = df["gk_distance_to_shooter"].isnull()
        df.loc[mask_nan_gk, "gk_distance_to_shooter"] = np.sqrt((120 - df.loc[mask_nan_gk, "x"])**2 + (40 - df.loc[mask_nan_gk, "y"])**2)
        df["num_opponents_in_shot_cone"] = df["num_opponents_in_shot_cone"].fillna(0)
        df["num_teammates_in_shot_cone"] = df["num_teammates_in_shot_cone"].fillna(0)
        df["total_players_in_shot_cone"] = df["num_opponents_in_shot_cone"] + df["num_teammates_in_shot_cone"]
        
        return df

    @classmethod
    def _clean_data(cls, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in ["effective_goal_width", "angle_from_center"]:
            if c in df.columns: df[c] = df[c].fillna(0)
        return df

def preprocess_pipeline(df):
    """Wrapper function to execute the full preprocessing pipeline"""
    df = filter_penalties(df)
    df = create_goal_flag(df)
    df = AdvancedFeatureEngineering.preprocess(df)
    df = handle_missing_values(df)
    df = add_dummy_features(df)
    return df

# ==========================================
# 2. Experiment Management and Saving
# ==========================================

def save_experiment(
    exp_name,
    features,
    results_df,
    coef_df,
    brier_custom,
    logloss_custom,
    y_test,
    xg_pred,
    base_dir="experiments"
):
    """
    Save results, calibration curves, and coefficient plots
    1 experiment = 1 folder
    """

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # ------------------------
    # 1. Save calibration curve
    # ------------------------
    prob_true, prob_pred = calibration_curve(y_test, xg_pred, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability (xG)")
    plt.ylabel("Observed goal rate")
    plt.title(f"Calibration Curve: {exp_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "calibration_curve.png"), bbox_inches='tight')
    plt.close()

    # ------------------------
    # 2. Save coefficient (importance) plot (★ modified part)
    # ------------------------
    # Create plot only if coef_df is not empty and contains 'coefficient' column
    if not coef_df.empty and 'coefficient' in coef_df.columns:
        # Sort by coefficient values
        plot_coef_df = coef_df.sort_values(by="coefficient", ascending=True)
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'blue' for x in plot_coef_df['coefficient']]
        plt.barh(plot_coef_df['feature'], plot_coef_df['coefficient'], color=colors)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel("Coefficient Value (Standardized)")
        plt.title(f"Feature Importance: {exp_name}")
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Save as image
        plt.savefig(os.path.join(exp_dir, "feature_importance.png"), bbox_inches='tight')
        plt.close()
    else:
        print("⚠️ Coefficient data (coef_df) is empty or has invalid format. Skipping feature importance plot.")

    # ------------------------
    # 3. Save all kinds of data and summaries
    # ------------------------
    # config.pkl
    config = {
        "exp_name": exp_name,
        "results": results_df,
        "features": features,
        "coefficient": coef_df,
        "timestamp": timestamp,
        "y_test": y_test,
        "xg_pred": xg_pred
    }
    with open(os.path.join(exp_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    # CSV
    results_df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)
    coef_df.to_csv(os.path.join(exp_dir, "coefficient.csv"), index=False)

    # summary.txt
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"brier: {brier_custom}\n")
        f.write(f"log loss: {logloss_custom}\n")
        f.write(f"Num features: {len(features)}\n")
        f.write(f"features: {features}\n")
        f.write(f"Num samples: {len(y_test)}\n")

    print(f"✅ Experiment saved to: {exp_dir}")
    if not coef_df.empty and 'coefficient' in coef_df.columns:
        print(f"📊 Figures saved: calibration_curve.png, feature_importance.png")
    else:
        print(f"📊 Figures saved: calibration_curve.png (Feature importance skipped)")

        
def plot_saved_performance_comparison(base_dir="experiments"):
    """
    Collect scores from all saved experiments and create a comparison plot
    Colors are unified by version
    """
    all_data = []
    config_paths = glob.glob(os.path.join(base_dir, "*/config.pkl"))

    for path in config_paths:
        with open(path, "rb") as f:
            conf = pickle.load(f)
            res = conf["results"]

            # Get experiment name and scores
            exp_name = conf.get("exp_name") or os.path.basename(os.path.dirname(path))
            # If you want to manage colors by pure experiment name without timestamp, you can process here

            all_data.append({
                "exp_name": exp_name,
                "brier": res["brier"].iloc[-1],
                "log_loss": res["log_loss"].iloc[-1]
            })

    if not all_data:
        print("No data found to display.")
        return

    # Convert to DataFrame and sort by Brier Score (higher = worse, easier to see improvement history)
    df_plot = pd.DataFrame(all_data).sort_values("brier", ascending=False)

    # --- Color settings: assign fixed colors per version ---
    # Generate different colors according to number of models
    num_models = len(df_plot)
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    # If you want to assign fixed colors to specific models, you can manage them with a dict

    # Adjust figure size according to number of models
    dynamic_width = max(10, num_models * 2.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(dynamic_width, 6))
    bar_width = 0.3  # "slimmer" as requested

    # --- 1. Brier Score (Left) ---
    bars1 = ax1.bar(df_plot["exp_name"], df_plot["brier"], color=colors, width=bar_width, alpha=0.8)
    ax1.set_title("Brier Score (Lower is Better)", fontsize=14, fontweight='bold')
    # Adjust lower limit to emphasize differences
    ax1.set_ylim(df_plot["brier"].min() * 0.98, df_plot["brier"].max() * 1.02)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")

    # --- 2. Log Loss (Right) ---
    # Use the same colors as left to keep version-color correspondence
    bars2 = ax2.bar(df_plot["exp_name"], df_plot["log_loss"], color=colors, width=bar_width, alpha=0.8)
    ax2.set_title("Log Loss (Lower is Better)", fontsize=14, fontweight='bold')
    # Adjust lower limit to emphasize differences
    ax2.set_ylim(df_plot["log_loss"].min() * 0.98, df_plot["log_loss"].max() * 1.02)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    # Add value labels
    def add_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.5f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(bars1, ax1)
    add_labels(bars2, ax2)

    plt.suptitle(f"Model Benchmarking: Metrics per Version", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

def compare_saved_experiments(base_dir="experiments", selected_exps=None):
    """
    Load data from each folder in base_dir and overlay calibration curves
    selected_exps: specify list if you want to filter by experiment name (prefix match, etc.)
    """
    # Adjust overall size. To make plot square, also give some margin in vertical direction
    # Around (12, 10) makes the plot large and square-ish
    plt.figure(figsize=(12, 10))
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Get all experiment folders
    exp_folders = sorted(glob.glob(os.path.join(base_dir, "*")))

    found_data = False
    for folder in exp_folders:
        config_path = os.path.join(folder, "config.pkl")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "rb") as f:
            data = pickle.load(f)

        # Filtering (if specified)
        if selected_exps and not any(sel in data["exp_name"] for sel in selected_exps):
            continue

        # Compute curve using saved data
        prob_true, prob_pred = calibration_curve(
            data["y_test"],
            data["xg_pred"],
            n_bins=10,
            strategy="uniform"
        )

        plt.plot(prob_pred, prob_true, marker="o", label=f"{data['exp_name']} ({data['timestamp']})")
        found_data = True

    if not found_data:
        print("No data found for comparison.")
        return

    plt.xlabel("Predicted probability (xG)")
    plt.ylabel("Observed goal rate")
    plt.title("Comparison of Calibration Curves")

    # Force the plot area to be square (1:1)
    plt.gca().set_aspect('equal')

    # Legend placement (outside right)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. Ensemble Calculation Class
# ==========================================
class EnsembleXGModel:
    def __init__(self, xgb_model, lr_model, scaler, xgb_weight):
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.scaler = scaler
        self.weight = xgb_weight
        self.feature_names_in_ = xgb_model.feature_names_in_
        self.classes_ = xgb_model.classes_

    def predict_proba(self, X):
        pred_xgb = self.xgb_model.predict_proba(X)[:, 1]
        X_scaled = self.scaler.transform(X)  # transform only
        pred_lr = self.lr_model.predict_proba(X_scaled)[:, 1]
        final_pred = (self.weight * pred_xgb) + ((1 - self.weight) * pred_lr)
        return np.column_stack([1 - final_pred, final_pred])
