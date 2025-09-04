import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("ceps_binary_ready.csv")

df_clean = df.dropna(subset=[
    "freq_binary", "parent_edu", "female", "grade9",
    "urbanicity_level", "has_siblings", "standard_living"
])

# Make sure your DataFrame has no missing values in the needed columns
df_clean = df_clean.dropna(subset=[
    "freq_binary", "parent_edu", "urbanicity_level",
    "female", "grade9", "has_siblings", "standard_living"
])

# Without interaction (better)
import statsmodels.formula.api as smf
df_clean["urbanicity_level"] = df_clean["urbanicity_level"].map({
    1: "urban",       # Urban Core
    2: "semiurban",   # Semi-Urban
    3: "rural"        # Rural
})

df_clean["urbanicity_level"] = pd.Categorical(
    df_clean["urbanicity_level"], categories=["rural", "semiurban", "urban"],
    ordered=False
)

df_clean["parent_edu"] = pd.Categorical(
    df_clean["parent_edu"], categories=["l", "m", "h"],
    ordered=False
)

# Fit the logistic regression model with main effects only
model_no_interaction = smf.logit(
    formula="""
        freq_binary ~ parent_edu + urbanicity_level
        + female + grade9 + has_siblings + standard_living
    """,
    data=df_clean
).fit()

print(model_no_interaction.summary())
print(f"BIC of the model: {model_no_interaction.bic:.2f}")

# Plot for the actual proportion of participation
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Step 1: Group by parent_edu and freq_binary with explicit observed=False
binary_counts = df_clean.groupby(["parent_edu", "freq_binary"], observed=False).size().unstack(fill_value=0)

# Step 2: Normalize to get proportions
binary_props = binary_counts.div(binary_counts.sum(axis=1), axis=0)

# Step 3: Reorder for logical x-axis
binary_props = binary_props.reindex(["l", "m", "h"])

# Step 4: Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Use soft blue tones
binary_props.plot(
    kind="bar", stacked=True, ax=ax,
    color=["#F4CCCC", "#E06666"],  # softer red and darker
    edgecolor="white"
)

# Step 5: Aesthetics
ax.set_title("Actual Binary Participation by Parental Education")
ax.set_xlabel("Parental Education Level")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.set_xticklabels(["Low", "Medium", "High"], rotation=0)

# Step 6: Custom legend
legend_elements = [
    Patch(facecolor="#F4CCCC", label="Non-or-Low Participant (0)"),
    Patch(facecolor="#E06666", label="Participant (1)")
]
ax.legend(handles=legend_elements, title="Participation")

plt.tight_layout()
plt.show()

