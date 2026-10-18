# Train Delay Prediction — End-to-End Predictive Analytics Project

Predicting the **severity of a train delay** (Low / Medium / High) from route, weather and timetable conditions, following the KDD / predictive-modelling stages: problem identification → data understanding → preprocessing → modelling → evaluation → interpretation.

## Open in Google Colab
1. Upload this repo to GitHub.
2. Open `train_delay_prediction.ipynb` in Colab (`File → Open notebook → GitHub`).
3. Run all cells. If the CSVs are not found, the notebook opens an upload dialog — upload both files from `data/`.

## Repository contents
| File | Description |
|---|---|
| `train_delay_prediction.ipynb` | Complete implementation (with outputs) |
| `data/train_delay_data.csv` | Real dataset (2,878 rows; 2,456 after removing duplicates) |
| `data/synthetic_train_delay_data.csv` | Synthetic dataset (5,000 rows) |
| `requirements.txt` | Python libraries used |

## Problem statement
Given the distance between stations, weather, train type and route congestion, predict whether a train will be **delayed by more than 15 minutes** (binary classification: *On-time* vs *Delayed*). Early warning of late trains helps operators re-plan crews, re-route traffic and inform passengers. A 3-class severity version (Low / Medium / High) is included as an extension.

## Datasets
Both files have the same 7 columns: `Distance Between Stations (km)`, `Weather Conditions`, `Day of the Week`, `Time of Day`, `Train Type`, `Route Congestion` and `Historical Delay (min)` (used to create the target).

* **Real dataset** – clear relationships between delay and distance, weather, train type and congestion.
* **Synthetic dataset** – same overall delay distribution, but the delay is statistically independent of the features (confirmed by correlation and Kruskal–Wallis tests).

## Methodology
1. **Data understanding & EDA** – shapes, missing values, duplicates, distributions, distance-vs-delay plots, mean delay by category, Kruskal–Wallis tests and a Cramér's V collinearity check.
2. **Preprocessing** – drop 422 duplicate rows in the real data *before* splitting (prevents leakage); create the `Delayed` / `On-time` target (> 15 min); drop `day` and `time_of_day` (no relationship with delay); one-hot encode categoricals and standardise distance inside a scikit-learn `Pipeline`; stratified 80/20 train/test split.
3. **Modelling** – majority-class baseline, Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting; 5-fold stratified cross-validation.
4. **Using both datasets** – six experiments (train/test on real, on synthetic, on the combination, and cross-dataset transfer).
5. **Tuning** – GridSearchCV (macro-F1, training data only) for Decision Tree, Gradient Boosting and Random Forest, including `class_weight` to handle the 71/29 class imbalance.
6. **Evaluation** – accuracy, **precision, recall and F1-score** (macro, weighted and per class), confusion matrix, nested cross-validation, permutation feature importance.

## Results (final model: tuned Random Forest)
| Evaluation | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Hold-out test set (492 trips), macro | 0.829 | 0.798 | 0.773 | 0.784 |
| Hold-out test set, 'Delayed' class only | 0.829 | 0.861 | 0.906 | 0.883 |
| Nested cross-validation, macro | 0.844 | 0.821 | 0.792 | 0.803 |

Majority-class baseline macro F1: 0.416.

## Key findings
* Weather and distance are the strongest predictors (distance works in bands: 201–300 km trips are rarely delayed, > 500 km trips almost always are). Day of week and time of day add nothing.
* In the real data, weather, train type and congestion are almost perfectly aligned with each other, so their individual importances cannot be cleanly separated.
* The synthetic dataset gives no-skill results and does not improve the model when merged with the real data, so it is used as a **negative control / robustness test** rather than as extra training data.
* The 3-class severity version reaches only ~0.70 accuracy / macro F1: inside each distance band the real delays are spread almost uniformly, so the exact severity band is largely random. The 15-minute delayed/on-time question is the one the data can answer reliably.

## Limitations
Small real dataset after de-duplication, collinear categorical features, no station/date identifiers, and a 15-minute threshold that is a design choice.

## How to run locally
```bash
pip install -r requirements.txt
jupyter notebook train_delay_prediction.ipynb
```
