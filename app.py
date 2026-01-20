import shutil
import os
import pandas as pd
from matplotlib.lines import Line2D
import streamlit as st
import traceback
import sys



# ==========================================
# 0. Page Settings (Highest Priority)
# ==========================================
st.set_page_config(page_title="WC 2022 xG Analyst", layout="wide")

# Safety wrapper for the entire app to display errors without crashing
try:
    import pandas as pd
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    import shap
    import ast
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.calibration import calibration_curve

    # Custom Styling
    st.markdown("""
        <style>
        .main > div {padding-top: 1rem;}
        .stPlotlyChart {background-color: #ffffff; border-radius: 5px;}
        </style>
        """, unsafe_allow_html=True)

    # ==========================================
    # 1. Class Definition (For model loading)
    # ==========================================
    try:
        import soccer_utils
        from soccer_utils import EnsembleXGModel
    except ImportError:
        # Fallback class if soccer_utils is missing
        class EnsembleXGModel:
            def __init__(self, xgb_model, lr_model, scaler, xgb_weight):
                self.xgb_model = xgb_model
                self.lr_model = lr_model
                self.scaler = scaler
                self.weight = xgb_weight

    # ==========================================
    # 2. Data Loading Functions
    # ==========================================
    @st.cache_resource
    def load_model():
        """Load the trained xG model from the models directory"""
        # Specify the file inside the 'model' folder using a relative path
        model_path = os.path.join('models', 'best_xg_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            st.error(f"❌ Error: '{model_path}' not found. Please check your 'models' folder.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Model loading error: {e}")
            st.stop()

    @st.cache_data
    def load_data():
        """Load and preprocess the World Cup shot data from the data directory"""
        # Specify the file inside the 'data' folder using a relative path
        data_path = os.path.join('data', 'wc_all_matches_scored_with_all_cols.csv')
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            st.error(f"❌ Error: '{data_path}' not found. Please check your 'data' folder.")
            st.stop()
        except Exception as e:
            st.error(f"❌ CSV loading error: {e}")
            st.stop()

        # Robust parsing logic for Freeze Frame data
        def parse_freeze_frame(x):
            if pd.isna(x) or x == "": return []
            if isinstance(x, list): return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    try:
                        # Clean broken formatting (e.g., double quotes) from CSV export
                        cleaned = x.strip().strip('"').strip("'").replace('""', '"')
                        return ast.literal_eval(cleaned)
                    except:
                        return []
            return []

        if 'shot_freeze_frame' in df.columns:
            df['shot_freeze_frame'] = df['shot_freeze_frame'].apply(parse_freeze_frame)

        # Fill missing metadata
        df['team_name'] = df['team_name'].fillna('Unknown')
        df['player_name'] = df['player_name'].fillna('Unknown')
        return df

    # Execute data loading
    df_wc = load_data()
    model_ensemble = load_model()
    model_xgb = model_ensemble.xgb_model if hasattr(model_ensemble, 'xgb_model') else model_ensemble

    st.sidebar.success("✅ System Ready (Full Version)")

    # Identify xG column name
    xg_col = 'xg' if 'xg' in df_wc.columns else 'shot_statsbomb_xg'

    # ==========================================
    # 3. Common: Create Non-Penalty Dataset
    # ==========================================
    # We exclude penalties for individual analysis (Deep Dive, Shot Map)
    # to match training data conditions
    df_no_pk = df_wc[df_wc['shot_type_name'] != 'Penalty'].copy()

    # ==========================================
    # 4. Drawing Functions & Label Generation
    # ==========================================
    def draw_pitch(ax, color='black'):
        """Function to draw a football pitch"""
        # Outer boundary
        plt.plot([0, 0], [0, 80], color=color, linewidth=1)
        plt.plot([0, 120], [80, 80], color=color, linewidth=1)
        plt.plot([120, 120], [80, 0], color=color, linewidth=1)
        plt.plot([120, 0], [0, 0], color=color, linewidth=1)
        # Boxes
        plt.plot([102, 102], [18, 62], color=color, linewidth=1)
        plt.plot([102, 120], [62, 62], color=color, linewidth=1)
        plt.plot([102, 120], [18, 18], color=color, linewidth=1)
        plt.plot([114, 114], [30, 50], color=color, linewidth=1)
        plt.plot([114, 120], [50, 50], color=color, linewidth=1)
        plt.plot([114, 120], [30, 30], color=color, linewidth=1)
        # Center line
        plt.plot([60, 60], [0, 80], color=color, linestyle='--', alpha=0.5)
        # Goal
        plt.plot([120, 120], [36, 44], color='red', linewidth=3)

        ax.set_aspect('equal')
        ax.axis('off')

    @st.cache_data
    def get_match_labels(df):
        match_labels = {}
        match_opponent_map = {}
        unique_matches = df['match_id'].unique()
        for mid in unique_matches:
            teams = df[df['match_id'] == mid]['team_name'].unique()
            if len(teams) >= 2:
                label = f"{teams[0]} vs {teams[1]}"
                match_labels[label] = mid
                match_opponent_map[mid] = {teams[0]: teams[1], teams[1]: teams[0]}
            elif len(teams) == 1:
                label = f"{teams[0]} (Single Data)"
                match_labels[label] = mid
                match_opponent_map[mid] = {teams[0]: "Unknown"}
        return match_labels, match_opponent_map

    match_labels_dict, match_opponent_map = get_match_labels(df_wc)

    # ==========================================
    # 5. Application UI
    # ==========================================
    st.sidebar.title("⚽ WC 2022 AI Analyst")
    page = st.sidebar.radio("Navigate", ["📊 Model Overview", "🌍 Tournament Analysis (Shot Map)", "⚽ Match Flow", "🔍 Deep Dive (XAI)"])

    # -----------------------------------------------------
    # PAGE 1: Model Overview
    # -----------------------------------------------------
    if page == "📊 Model Overview":
        st.title("📊 Model Overview")
        st.write("Evaluation of model fit against World Cup data")

        # Evaluation uses Non-Penalty data (matching training conditions)
        y_true, y_prob = df_no_pk['is_goal'], df_no_pk[xg_col]

        c1, c2 = st.columns(2)
        c1.metric("Brier Score", f"{brier_score_loss(y_true, y_prob):.4f}")
        c2.metric("Log Loss", f"{log_loss(y_true, y_prob):.4f}")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Calibration Curve")
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(prob_pred, prob_true, marker='o', label='Model', color='#1f77b4')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
            ax.set_xlabel("Predicted Probability (xG)")
            ax.set_ylabel("Actual Goal Rate")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.subheader("Feature Importance")
            if hasattr(model_xgb, 'feature_importances_'):
                f_names = getattr(model_xgb, 'feature_names_in_', [f"F{i}" for i in range(len(model_xgb.feature_importances_))])
                feat_imp = pd.Series(model_xgb.feature_importances_, index=f_names).sort_values().tail(10)

                fig, ax = plt.subplots(figsize=(6, 5))
                feat_imp.plot(kind='barh', ax=ax, color='#1f77b4')
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig)


    # ---------------------------------------------------------
    # PAGE 2: Tournament Analysis
    # ---------------------------------------------------------
    elif page == "🌍 Tournament Analysis (Shot Map)":
        st.title("🌍 Tournament Analysis")
        st.info("Note: Data on this page is aggregated excluding Penalties (Non-Penalty).")

        tab1, tab2 = st.tabs(["Player Shot Map", "Performance Stats"])

        with tab1:
            st.subheader("📍 Player Shot Map")
            c1, c2 = st.columns(2)

            # Selection logic: Pick team first, then pick player
            teams = sorted(df_no_pk['team_name'].unique())
            s_team = c1.selectbox("Team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
            players = sorted(df_no_pk[df_no_pk['team_name'] == s_team]['player_name'].unique())
            s_player = c2.selectbox("Player", players, index=players.index("Lionel Andrés Messi Cuccittini") if "Lionel Andrés Messi Cuccittini" in players else 0)

            player_df = df_no_pk[df_no_pk["player_name"] == s_player].copy()

            if not player_df.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                draw_pitch(ax)
                ng = player_df[player_df["is_goal"]==0]
                g = player_df[player_df["is_goal"]==1]

                ax.scatter(ng["x"], ng["y"], alpha=0.6, color="blue", label="No Goal", s=40, marker="x")

                if not g.empty:
                    ax.scatter(g["x"], g["y"], s=g[xg_col]*600+50, color="red", edgecolors="black", label="Goal", alpha=0.8, zorder=5)

                ax.set_title(f"Shot Map: {s_player}", fontsize=14)
                ax.set_xlim(60, 125); ax.set_ylim(0, 80)
                ax.invert_yaxis() # Invert Y-axis for standard pitch view
                ax.legend(loc='lower left', fontsize='small')
                st.pyplot(fig, use_container_width=False)

                st.metric("Total Goals", int(player_df['is_goal'].sum()))
                st.metric("Total xG", f"{player_df[xg_col].sum():.4f}")
            else:
                st.warning("No data found.")

        with tab2:
            st.subheader("Goals vs xG (Ranking)")
            import plotly.express as px

            p_stats = df_no_pk.groupby(['player_name', 'team_name']).agg({'is_goal':'sum', xg_col:'sum', 'id':'count'}).reset_index()
            p_stats['Diff'] = p_stats['is_goal'] - p_stats[xg_col]
            fig = px.scatter(p_stats[p_stats['id']>=3], x=xg_col, y='is_goal', hover_data=['player_name'], color='Diff', color_continuous_scale='Portland', title="Goals vs xG")

            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

            # Add y=x diagonal line
            max_val = max(p_stats['is_goal'].max(), p_stats[xg_col].max()) + 1
            fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))

            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # PAGE 3: Match Flow
    # -----------------------------------------------------
    elif page == "⚽ Match Flow":
        st.title("⚽ Match Flow")
        st.info("Note: Data on this page includes Penalties (With Penalty).")

        # Select match
        s_match = st.selectbox("Match", list(match_labels_dict.keys()))
        mid = match_labels_dict[s_match]

        # Use full df_wc (including PKs)
        df_m = df_wc[(df_wc['match_id'] == mid) & (df_wc['period'] <= 4)].sort_values(['period','minute'])
        teams = df_m['team_name'].unique()

        cols = st.columns(2)
        for i, team in enumerate(teams):
            if i < 2:
                t_goals = df_m[(df_m['team_name']==team) & (df_m['is_goal']==1)].shape[0]
                t_xg = df_m[df_m['team_name']==team][xg_col].sum()
                cols[i].metric(team, f"{t_goals} Goals", f"xG: {t_xg:.4f}")

        # Cumulative xG Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#1f77b4', '#ff7f0e']

        for i, t in enumerate(teams):
            dft = df_m[df_m['team_name']==t]

            xg_cum = [0] + dft[xg_col].cumsum().tolist() + [dft[xg_col].sum()]
            mins = [0] + dft['minute'].tolist() + [df_m['minute'].max()+2]

            ax.step(mins, xg_cum, where='post', label=t, color=colors[i%2], linewidth=2)

            # Mark goals on the line
            for _, g in dft[dft['is_goal']==1].iterrows():
                c_val = dft[dft.index<=g.name][xg_col].sum()
                ax.scatter(g['minute'], c_val, s=100, facecolors='white', edgecolors=colors[i%2], zorder=5, linewidth=2)
                ax.text(g['minute'], c_val+0.1, "⚽", ha='center')

        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        st.pyplot(fig)


    # -----------------------------------------------------
    # PAGE 4: Deep Dive (XAI)
    # -----------------------------------------------------
    elif page == "🔍 Deep Dive (XAI)":
        st.title("🔍 Shot Deep Dive")
        st.info("Note: Data on this page excludes Penalties (Non-Penalty).")

        c1, c2 = st.columns(2)
        teams = sorted(df_no_pk['team_name'].unique())
        team = c1.selectbox("Team", teams)

        players = sorted(df_no_pk[df_no_pk['team_name']==team]['player_name'].unique())
        player = c2.selectbox("Player", players)

        shots = df_no_pk[(df_no_pk['team_name']==team) & (df_no_pk['player_name']==player)].copy()

        if len(shots) > 0:
            def get_shot_label(row):
                opp = match_opponent_map.get(row['match_id'], {}).get(row['team_name'], "Unknown")
                res = "⚽" if row['is_goal'] else "❌"
                return f"{res} vs {opp} | {row['minute']}min | xG: {row[xg_col]:.4f}"

            shots['label'] = shots.apply(get_shot_label, axis=1)
            sel_lbl = st.selectbox("Situation", shots['label'])
            row = shots[shots['label']==sel_lbl].iloc[0]

            st.markdown("---")
            col_vis, col_shap = st.columns([1.3, 1])

            # --- Freeze Frame Plot ---
            with col_vis:
                st.subheader("📸 Freeze Frame")
                fig, ax = plt.subplots(figsize=(8, 6))
                draw_pitch(ax)
                ff = row['shot_freeze_frame']
                has_players = False
                if isinstance(ff, list) and len(ff) > 0:
                    has_players = True
                    for p in ff:
                        if isinstance(p, dict) and 'x' in p and 'y' in p:
                            is_teammate = p.get('teammate', False)
                            is_gk = (p.get('position_name', '') == 'Goalkeeper')
                            # Color logic: Teammate = Blue, Opponent = Red, GK = Black Square
                            if is_teammate: c, m, a, s = '#1f77b4', 'o', 0.9, 9
                            else: c, m, a, s = ('black', 's', 1.0, 11) if is_gk else ('#d62728', 'o', 0.9, 9)
                            ax.plot(p['x'], p['y'], marker=m, color=c, markeredgecolor='white', markersize=s, alpha=a, zorder=5)

                # Shooter (Gold Star)
                ax.plot(row['x'], row['y'], '*', color='gold', markersize=18, markeredgecolor='black', zorder=10, label='Shooter')

                # Shot Angle visualization
                ax.plot([row['x'], 120], [row['y'], 36], color='gold', alpha=0.2, linestyle='--')
                ax.plot([row['x'], 120], [row['y'], 44], color='gold', alpha=0.2, linestyle='--')

                # Legend
                legend_elements = [
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', label='Shooter', markersize=12, markeredgecolor='k'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Teammate', markersize=9),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Opponent', markersize=9),
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='black', label='Opponent GK', markersize=10),
                ]
                ax.legend(handles=legend_elements, loc='lower left', frameon=True)

                ax.set_xlim(60, 122); ax.set_ylim(0, 80)
                ax.invert_yaxis()
                st.pyplot(fig)

                if not has_players:
                    st.warning("⚠️ No freeze frame player data found.")

  
            # ==========================================
            # Ensemble SHAP Implementation
            # ==========================================
            with col_shap:
                st.subheader("🧠 SHAP Explanation")

                # 1. Preparation: Retrieve models and scaler
                # (From the EnsembleXGModel class instance)
                final_xgb_model = model_ensemble.xgb_model
                final_lr_model = model_ensemble.lr_model
                final_scaler = model_ensemble.scaler
                w = model_ensemble.weight # Weight for XGBoost

                # 2. Create Background Data (Using the full WC dataset)
                # * df_no_pk is the full dataset already loaded in the app (excluding penalties)
                # Align the feature order with the XGBoost model
                X_background = df_no_pk[final_xgb_model.feature_names_in_].apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Scale the data for Logistic Regression (using the saved scaler)
                X_background_scaled = final_scaler.transform(X_background)

                # 3. Prepare Target Data (The specific shot selected)
                # 'row' is the data for the single selected shot
                X_target = row[final_xgb_model.feature_names_in_].to_frame().T.apply(pd.to_numeric, errors='coerce').fillna(0)
                X_target_scaled = final_scaler.transform(X_target)

                # -------------------------------------------------
                # 4. SHAP values for XGBoost
                # -------------------------------------------------
                explainer_xgb = shap.TreeExplainer(final_xgb_model)
                shap_values_xgb = explainer_xgb(X_target)

                # -------------------------------------------------
                # 5. SHAP values for Logistic Regression (LinearExplainer)
                # -------------------------------------------------
                # * Pass the scaled background data (full WC dataset) here for accurate calculation
                explainer_lr = shap.LinearExplainer(final_lr_model, X_background_scaled, feature_perturbation="interventional")
                shap_values_lr = explainer_lr(X_target_scaled)

                # -------------------------------------------------
                # 6. Ensemble (Weighted Average)
                # -------------------------------------------------
                # Combine the SHAP values and base values using the model weights
                ensemble_values = (w * shap_values_xgb.values) + ((1 - w) * shap_values_lr.values)
                ensemble_base_values = (w * shap_values_xgb.base_values) + ((1 - w) * shap_values_lr.base_values)

                # 7. Create SHAP "Explanation" Object for Display
                shap_explanation = shap.Explanation(
                    values=ensemble_values[0],       # Extract [0] as it's a single row
                    base_values=ensemble_base_values[0],
                    data=X_target.iloc[0],           # Use original values (meters, degrees) for readability
                    feature_names=final_xgb_model.feature_names_in_
                )

                # 8. Plot
                fig_s, ax_s = plt.subplots(figsize=(5, 6))
                shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                st.pyplot(fig_s)

except Exception:
    st.error("🚨 An error occurred during app execution")
    st.code(traceback.format_exc())

