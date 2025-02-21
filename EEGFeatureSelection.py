import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

def calculate_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    """Compute the confidence interval for the R² score using bootstrapping."""
    r2_scores = []
    n_samples = len(y_true)
    alpha = (100 - ci) / 2
    np.random.seed(42)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        r2 = r2_score(y_true_boot, y_pred_boot)
        r2_scores.append(r2)
    lower = np.percentile(r2_scores, alpha)
    upper = np.percentile(r2_scores, 100 - alpha)
    return lower, upper

# 1️⃣ Load dataset
file_path = "/data/alldata_51.xlsx"
df = pd.read_excel(file_path)

# 2️⃣ Define relevant columns
# Selecting EEG features based on column name patterns
eeg_features = [
    col for col in df.columns if col.startswith(("Total", "IndExpVar", "MeanDuration", "MeanOccurrence", 
                                                 "Coverage", "MeanGFP", "SelfTransitions", "OrgTM"))
]
# Define target variables for prediction
targets = ["Anxiety", "SeparationAnxiety", "ACC", "real_RT"]
# Define output directory and create it if it does not exist
output_dir = "/MLMLML2"
os.makedirs(output_dir, exist_ok=True)

# 3️⃣ Perform analysis separately for each group
results = []# Store model performance metrics
feature_importance = []# Store the best model results
best_results = []

for group in [1, 2]:  # Filter data for the current group and drop missing values in relevant columns
    df_group = df[df["Group"] == group].dropna(subset=targets + eeg_features)
    
    # Normalize EEG features
    scaler = StandardScaler()
    df_group[eeg_features] = scaler.fit_transform(df_group[eeg_features])
    
    for target in targets:
        X = df_group[eeg_features]
        y = df_group[target]
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models to be tested
        models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "LightGBM": LGBMRegressor(num_leaves=10, min_data_in_leaf=5, max_depth=5, random_state=42)
        }

        best_r2 = -np.inf  # Track the highest R² score
        best_model_name = None
        best_y_test = None
        best_y_pred = None

        for name, model in models.items():
             # Train the model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Evaluate performance
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results.append({"Group": group, "Target": target, "Model": name, "R2 Score": r2, "MSE": mse})

            # Update the best model if it has the highest R² score
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_y_test = y_test.to_numpy()
                best_y_pred = y_pred.copy()

        # Compute confidence interval for the best model
        if best_model_name:
            ci_lower, ci_upper = calculate_ci(best_y_test, best_y_pred)
            best_results.append({
                "Target": target,
                "Group": group,
                "Best Model": best_model_name,
                "R²": round(best_r2, 2),
                "95% CI": f"[{ci_lower:.2f}, {ci_upper:.2f}]"
            })
            
            # 📌 Residual plot
            plt.figure(figsize=(6, 4))
            residuals = best_y_test - best_y_pred
            sns.scatterplot(x=best_y_pred, y=residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title(f"Residual Plot ({best_model_name} - {target})")
            plt.savefig(f"{output_dir}/residual_plot_{group}_{target}.png")
            plt.close()
            
            # 📌 Prediction error distribution plot
            plt.figure(figsize=(6, 4))
            sns.histplot(residuals, bins=20, kde=True)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"Error Distribution ({best_model_name} - {target})")
            plt.savefig(f"{output_dir}/error_distribution_{group}_{target}.png")
            plt.close()

# 4️⃣ Save results to CSV files
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/newmodel_performance.csv", index=False)

best_results_df = pd.DataFrame(best_results)
best_results_df.to_csv(f"{output_dir}/newbest_model_r2_summary.csv", index=False)

print(f"Analysis complete. Results have been saved to {output_dir}")