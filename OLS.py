import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf

# Load pre-processed data
df = pd.read_csv("ceps_ols_ready.csv")

'''
# Without interaction

# The first time
# Define the dependent variable (Y)
# y = df["frequency_num"]
# Define the independent variables (X)
# X = df[[
#     "parent_edu", "female", "grade9", "urbanicity_level", "has_siblings", "standard_living"
# ]]
# Add a constant term (intercept)
# X = sm.add_constant(X)
# Fit OLS model
# model = sm.OLS(y, X).fit()
# Print summary
# print(model.summary())

# The second time: region cannot be treated as continuous numbers
model = smf.ols(
    'frequency_num ~ parent_edu + female + grade9 + C(urbanicity_level) + has_siblings + standard_living',
    data=df
).fit()
print(model.summary())
'''

'''
# With interaction
# Create interaction term
df["parentedu_urban"] = df["parent_edu"] * df["urbanicity_level"]
import statsmodels.formula.api as smf
model = smf.ols(
    'frequency_num ~ parent_edu * C(urbanicity_level) + female + grade9 + has_siblings + standard_living',
    data=df
).fit()
print(model.summary())
'''

# F-test
df_clean = df.dropna(subset=['frequency_num', 'parent_edu', 'urbanicity_level', 'female', 'grade9', 'has_siblings', 'standard_living'])

# Then fit both models on df_clean
model_main = smf.ols('frequency_num ~ parent_edu + C(urbanicity_level) + female + grade9 + has_siblings + standard_living', data=df_clean).fit()
model_interaction = smf.ols('frequency_num ~ parent_edu * C(urbanicity_level) + female + grade9 + has_siblings + standard_living', data=df_clean).fit()

# Compare 
rss_main = model_main.ssr
rss_interaction = model_interaction.ssr

df_main = model_main.df_resid
df_interaction = model_interaction.df_resid

numerator = (rss_main - rss_interaction) / (df_main - df_interaction)
denominator = rss_interaction / df_interaction

F = numerator / denominator

print("F-statistic:", F)

from scipy.stats import f

p_value = 1 - f.cdf(F, df_main - df_interaction, df_interaction)
print("p-value:", p_value)
