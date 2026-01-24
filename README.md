# ⚽ FreezeFrame xG: Context-Aware Expected Goals Model

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)
![StatsBomb](https://img.shields.io/badge/Data-StatsBomb_Open_Data-red.svg)

> **"Beyond Location"** — A high-precision Expected Goals (xG) model that goes beyond simple location data by incorporating "Freeze Frame" context such as defensive pressure and visual field geometry.
> Features include geometric feature engineering using Polygons and XAI (SHAP) for factor analysis.

## 🔗 Live Demo
**👉 [Launch the Web App]https://soccer-xg-analysis-f6pgtnnzixwnbfbavbfrrk.streamlit.app/**
*(Running on Streamlit Cloud / Data: FIFA World Cup 2022)*

---

## 1. Executive Summary

### 🎯 Challenge
The goal was not to create a simple xG model based solely on shot distance and angle, but to build a high-precision model that incorporates **"contextual information of the match."**
Furthermore, the project aimed to use this model to quantitatively analyze match flow and individual player shooting abilities.

Key contextual factors addressed:
* **Defensive Pressure:** Was the shooter under pressure?
* **Shot Blocking:** How many defenders were blocking the goal line?
* **GK Position:** Was the goalkeeper in position or out of position?
* **Vision:** How wide did the goal appear to the shooter, considering obstacles?

### 🛠 Methodology & Innovations
* **Ensemble Learning:** Utilized an ensemble of Logistic Regression (for interpretability) and XGBoost (for capturing non-linearities).
* **Geometric Feature Engineering:** Used the `Shapely` library to calculate the number of opponents within the "Shot Cone."
* **XAI (Explainable AI):** implemented SHAP values to visualize the "factors" behind the expected goal value.
* **Web Application:** Developed a Streamlit app to visualize match momentum and individual performance using World Cup data.

### 🏆 Final Results
In this project, we validated not only the accuracy scores but also the **Calibration Curve** to ensure that the predicted probabilities correctly reflect reality.

| Model | Brier Score (↓) | Log Loss (↓) | Note |
| :--- | :---: | :---: | :--- |
| Logistic Regression | 0.07353 | 0.26117 | Baseline |
| XGBoost (Tuned) | 0.07370 | 0.26160 | Non-linear features |
| **Ensemble (Final)** | **0.07330** | **0.25990** | **Best Model** |

---

## 2. Data Source

This project utilizes **StatsBomb Open Data**.

* **For Training:** England Premier League (2015/2016)
* **For Application/Validation:** FIFA World Cup 2022

*Note: The World Cup dataset had a slightly different column structure compared to the Premier League data, so preprocessing was performed to align the schemas.*

---

## 3. Analysis Flow

The raw Event Data provided by StatsBomb contains pitch coordinates, but **it does not contain any model-ready features** such as "distance to goal" or "shot angle."
Therefore, I started with a baseline model and iteratively improved performance by adding features based on specific hypotheses.

### 🔄 Development Steps
1.  **Data Acquisition & Preprocessing**
2.  **Create BASE_FEATURES** → Train/Validate Logistic Regression
3.  **Add CONTEXT_FEATURES** → Train/Validate Logistic Regression
4.  **Add ADVANCED_GEOMETRY_FEATURES** → Train/Validate Logistic Regression
5.  **Add ASSIST_FEATURES** → Train/Validate Logistic Regression
6.  **Add SHOT_BLOCK_FEATURES** → Train/Validate Logistic Regression
    * *(Confirmed the effectiveness of features up to this point)*
7.  **Train/Validate XGBoost with ALL Features** (Cross-Validation)
8.  **Hyperparameter Tuning** (for both LR & XGBoost)
9.  **Create Ensemble Model** (Final Validation)

---

## 4. Feature Engineering (Technical Deep Dive)

The most significant factor in improving accuracy in this project was the **creation of complex features**.
I focused heavily on calculating distance, angles, and polygons directly from coordinate data.

### 📐 Feature Logic
* **`shot_distance`**: Calculated Euclidean distance to the center of the goal using the Pythagorean theorem.
* **`shot_angle`**: Calculated the viewing angle using the vector dot product and magnitude between the shooter and the two goalposts.
* **`effective_goal_width`**: The physical goal width is constant, but it appears narrower when viewed from an angle. I used trigonometry to calculate the "apparent goal width" visible to the player.
* **`total_players_in_shot_cone`**: Used **Polygons** (via Shapely) to create a triangle connecting the shooter and the two goalposts (Shot Cone). Using the Point logic, I counted the number of players located within that triangle.
    * *Note: This "Shot Block Feature" contributed significantly to score improvement.*

### 📝 Feature Groups
Features were managed in groups to verify model improvement step-by-step.

#### 1. 📍 BASE_FEATURES
Basic variables based only on physical coordinates.

| Feature Name | Description | Note |
| :--- | :--- | :--- |
| `shot_distance` | Distance to goal center | Closer = Higher probability (Negative Coef) |
| `shot_angle` | Angle of the goal (Vision) | Frontal view = Higher probability (Positive Coef) |

#### 2. 🧠 CONTEXT_FEATURES
Contextual information that cannot be determined by location alone.

| Feature Name | Description | Note |
| :--- | :--- | :--- |
| `under_pressure` | Was the shooter under pressure? | True = Lower probability |
| `body_Head` | Header | Lower probability than foot shots |
| `body_Left/Right Foot` | Which foot was used? | |
| `shoot_type` | Open Play, Free Kick, Corner, etc. | |
| `pattern_From Counter` | Shot from a counter-attack | Tendency to be a big chance |
| `shoot_technic` | Volley, Lob, Diving Header, etc. | Represents difficulty |

#### 3. 📐 GEOMETRY_FEATURES (Freeze Frame)
Advanced features calculated using the **Shapely** library.

| Feature Name | Description | Note |
| :--- | :--- | :--- |
| `gk_distance_to_shooter` | Linear distance to GK | Real distance to GK, not goal line |
| `num_opponents_in_shot_cone` | Opponents in Shot Cone | Block factor. More players = Significant drop |
| `num_teammates_in_shot_cone` | Teammates in Shot Cone | Blind factor |
| `effective_goal_width` | Visible Goal Width | Apparent width considering obstacles |
| `distance_angle_interaction` | Distance × Angle | Emphasizes "Close range & High angle" |
| `distance_squared` | Distance squared | Captures non-linear effect of distance |

#### 4. ⚽ ASSIST_FEATURES
Information regarding the pass leading to the shot.

| Feature Name | Description | Note |
| :--- | :--- | :--- |
| `has_assist` | Was there an assist? | Distinguish from solo runs/rebounds |
| `pass_length` | Distance of the pass | |
| `is_cross` | Shot from a cross | High difficulty |
| `is_cutback` | Shot from a cutback | High probability chance |
| `is_through_ball` | Shot from a through ball | High chance of 1-on-1 |
| `is_pass_high` | High ball pass | Increases difficulty (Volley/Head) |

---

## 5. Experiments & Results

### Hypothesis Testing Process
I started with a simple model and verified how accuracy changed as I added features based on specific hypotheses.

* **Phase 1: Location Only** (Base)
    * *Hypothesis:* How well can we predict using only basic physical geometry?
* **Phase 2: Context Features** (+ Pressure, Body part)
    * *Hypothesis:* Information like "Free or not" and "Header or Foot" should reduce uncertainty.
* **Phase 3: Advanced Geometry** (+ Goal Width, Interaction)
    * *Hypothesis:* Will advanced geometric features like apparent goal width improve the score?
* **Phase 4: Assist Features** (+ Pass info)
    * *Hypothesis:* The quality of the final pass (e.g., floating ball) changes the difficulty.
* **Phase 5: Shot Block Features** (+ Shot Cone)
    * *Hypothesis:* Information on whether the shot course is blocked should allow for much higher precision.

### Validation Scores (Evolution)
Experimental results showed that **Log Loss (Uncertainty)** and **Brier Score (Error)** steadily decreased with each addition of features. The improvement was particularly notable when `Context` and `Shot Block` features were added.

| Model Phase | Brier Score (↓) | Log Loss (↓) |
| :--- | :---: | :---: |
| **Base Features** | 0.07775 | 0.27601 |
| **+ Context Features** | 0.07440 | 0.26315 |
| **+ Geometry Features** | 0.07431 | 0.26250 |
| **+ Assist Features** | 0.07394 | 0.26060 |
| **+ Shot Block Features** | **0.07297** | **0.25753** |

### Final Model Selection
Based on the results above, I decided to use all features. After performing Cross-Validation (CV) and Hyperparameter Tuning, I compared the final candidate models.

| Model | Brier Score | Log Loss | Verdict |
| :--- | :---: | :---: | :--- |
| Logistic Regression | 0.07353 | 0.26117 | |
| XGBoost | 0.07370 | 0.26160 | |
| **Ensemble** | **0.07330** | **0.25990** | **Best** |

Since the Ensemble model showed the best performance in both Brier Score and Log Loss, and the **Calibration Curve depicted a nearly diagonal line**, I selected it as the Best Model.

---

## 6. Modeling Strategy & Evaluation Metrics


### Models Used

1. **Logistic Regression:**
* Adopted as a baseline for its interpretability (easy to understand coefficient impact).


2. **XGBoost (Gradient Boosting):**
* Adopted to capture complex **non-linear interactions** between variables (e.g., "Distance is close, but there are many defenders").



### Validation Strategy

To ensure the model's generalizability to unseen games, strict validation protocols were implemented:

* **GroupKFold Cross-Validation:**
* Standard random splitting (KFold) causes **Data Leakage** in soccer analytics because events within the same match are highly correlated (e.g., defensive intensity, weather, scoreline effects).
* This project adopted **GroupKFold** grouped by `match_id`. This ensures that the training and test sets contain completely distinct matches, simulating a real-world scenario where the model predicts outcomes for future games.



### Evaluation Metrics

Soccer shot data is **Imbalanced Data** where "only about 1 in 10 shots results in a goal." Therefore, simple Accuracy is not an appropriate metric.
This project prioritized the **"Reliability of Probability"** using the following metrics:

* **Log Loss:**
* Evaluates the "uncertainty" of the model. It imposes a heavy penalty on "overconfident mistakes" (e.g., predicting 100% goal and missing), encouraging honest probability outputs.


* **Brier Score:**
* The mean squared error of the predicted probability and the actual result. Closer to 0 indicates higher accuracy.


* **Calibration Curve:**
* Visualizes "If the model predicts 30%, do 30% of those shots actually go in?" This confirms the reliability of the model.

---

## 7. Web Application Features

I built a Streamlit application to analyze the 2022 World Cup data using the created Ensemble model.

### 📊 Key Features
1.  **xG Flow Chart (Match Momentum):**
    * Visualizes how the xG of both teams changed over time in a match. It reveals "which team was dominant," which cannot be understood from shot counts alone.
2.  **Player Performance:**
    * **xG vs Goals:** Plots how much a player was involved in big chances vs. how much finishing ability they demonstrated.
    * **Shot Map:** Plots shot locations for each player on the pitch.
3.  **Freeze Frame & SHAP Visualization:**
    * **Context Visualization:** Plots the positions of all players at the moment of the shot to visually confirm difficulty.
    * **SHAP Values:** Quantitatively displays the factors (e.g., "GK was close," "High density of opponents") explaining *why* the model output that specific xG value.

---

## 8. Conclusion

* Incorporating high-level context information and geometric information (Freeze Frame) definitively improves model accuracy.
* Ensembling Linear models (LR) and Non-linear models (XGB) creates a highly accurate and stable prediction model that cannot be achieved by either model alone.

---

## 9. Repository Structure

```text
football-xg-portfolio/
│
├── app.py                     # Streamlit Application Entry Point
├── EN_soccer_utils.py         # Feature Engineering & Geometry Utils
├── requirements.txt           # Dependencies
│
├── models/                    # Trained Models
│   └── best_xg_model.pkl      # Ensemble Model
│
├── notebooks/                 # Analysis Process
│   ├── EN_data_create_file.ipynb              # Data Preprocessing
│   ├── EN_soccer_tactical_analysis...ipynb    # Model Training & Experiments
│   └── EN_04_predict_wc_data.ipynb            # Inference on WC 2022 data
│
└── experiments/               # Experiment Logs & Artifacts
    ├── LogisticRegression_Tuned_version
    ├── XGBoost_Tuned_version
    └── Ensemble_version
```

---


## 10. Tech Stack

* **Language:** Python 3.12
* **Data Analysis:** `pandas`, `numpy`
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Feature Engineering:** `shapely` (Geometric calculations)
* **Visualization:** `matplotlib`, `seaborn`, `shap` (XAI)
* **Web Application:** `streamlit`
* **Data Source:** `statsbombpy` (StatsBomb Open Data)

---

## License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.




