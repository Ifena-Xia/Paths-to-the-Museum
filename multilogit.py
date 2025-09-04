# Without interaction (better)
import pandas as pd
import statsmodels.api as sm

# === 1. Load your dataset ===
df = pd.read_csv("ceps_mlogit_ready.csv")

# === 2. Ensure the dependent variable is clean categorical ===
df["participation_level"] = pd.Categorical(df["participation_level"], categories=["low", "moderate", "high"])
y = df["participation_level"].cat.codes  # 0 = low, 1 = moderate, 2 = high

# === 3. Create the design matrix (X) ===
X = df[[
    "parent_edu_m", "parent_edu_h",
    "female", "grade9", "has_siblings",
    "standard_living"
]]

# Add dummy variables for urbanicity (drop_first to avoid collinearity)
urban_dummies = pd.get_dummies(df["urbanicity_level"], prefix="urbanicity", drop_first=True)

# Concatenate them
X = pd.concat([X, urban_dummies], axis=1)

# Add constant
X = sm.add_constant(X)

# === 4. Now the brute-force safety check: cast everything ===
X = X.copy().astype("float64")  # <== This kills all object-type issues
y = y.astype("int64")

# === 5. Drop rows with any NaN (just in case) ===
mask = X.notnull().all(axis=1) & pd.notnull(y)
X = X[mask]
y = y[mask]

# === 6. Fit the model ===
model = sm.MNLogit(y, X).fit()
print(model.summary())
print(f"BIC of the model: {model.bic:.2f}")


# Separate plots
# Avarage data: real data look
# Predict probabilities for all data

pred_probs = model.predict(X)

# Add them to the dataframe (make a copy to avoid changing the original)
df_plot = df.loc[X.index].copy()
df_plot[["prob_low", "prob_mod", "prob_high"]] = pred_probs

import matplotlib.pyplot as plt
import seaborn as sns

# Create a new column to label the category based on dummies
def recover_parent_edu(row):
    if row["parent_edu_h"] == 1:
        return "h"
    elif row["parent_edu_m"] == 1:
        return "m"
    else:
        return "l"

df_plot["parent_edu"] = df_plot.apply(recover_parent_edu, axis=1)

# Melt for Seaborn
df_melted = df_plot.melt(
    id_vars="parent_edu", 
    value_vars=["prob_low", "prob_mod", "prob_high"],
    var_name="level", value_name="pred_prob"
)
# Melt() transforms a wide table into a long table
# Now Seaborn can easily:
# Use parent_edu on the x-axis
# Use pred_prob on the y-axis
# Color lines by level

# Clean level names
df_melted["level"] = df_melted["level"].map({
    "prob_low": "Low",
    "prob_mod": "Moderate",
    "prob_high": "High"
})

# Define colors
soft_purple = "#9c89b8"
warm_yellow = "#f6c453"

# Plot
# Get actual proportion per group
actual_dist = df_plot.groupby(["parent_edu", "participation_level"], observed=False).size().unstack().fillna(0)
actual_props = actual_dist.div(actual_dist.sum(axis=1), axis=0)

# Reorder index to l, m, h
actual_props = actual_props.reindex(["l", "m", "h"])

# Plot
ax = actual_props[["low", "moderate", "high"]].plot(
    kind="bar", stacked=True, figsize=(10, 6),
    color=[soft_purple, "#c3aed6", "#e0bbe4"]
)

plt.title("Actual Participation Distribution by Parental Education")
plt.ylabel("Proportion")
plt.xlabel("Parental Education Level")
plt.xticks(ticks=[0, 1, 2], labels=["Low", "Medium", "High"], rotation=0)

# Pretty legend
ax.legend(title="Participation Level", loc="upper right")

plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Clean model curve
# Fix x-axis order
df_melted["parent_edu"] = pd.Categorical(
    df_melted["parent_edu"],
    categories=["l", "m", "h"],
    ordered=True
)

# Contrasting yellows
palette_yellows = ["#f6c453", "#e5b032", "#c99700"]

# Plot with better x-axis order and contrast
plt.figure(figsize=(10, 6))
sns.pointplot(
    x="parent_edu", y="pred_prob", hue="level",
    data=df_melted, linestyles="--", markers="o",
    palette=palette_yellows
)

plt.title("Model-Predicted Probability of Cultural Participation")
plt.ylabel("Predicted Probability")
plt.xlabel("Parental Education Level")
plt.ylim(0, 1)
plt.legend(title="Participation Level", loc="upper right")
plt.tight_layout()
plt.show()
