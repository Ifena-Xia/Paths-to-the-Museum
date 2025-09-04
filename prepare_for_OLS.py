import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_json("ceps_final_translated_with_years.json")
df = df[df["living_area"] != "Other"]
# 1) Encode frequency (Never = 1, More than once a week = 6)
freq_map = {
    "Never": 1,
    "Once a year": 2,
    "Once every half year": 3,
    "Once a month": 4,
    "Once a week": 5,
    "More than once a week": 6
}
df["frequency_num"] = df["frequency"].map(freq_map)

# 2) Select higher of mother/father education in years
df["parent_edu"] = df[["mo_edu_y", "fa_edu_y"]].max(axis=1)

# 3) Create dummy variables
# Gender: female = 1, male = 0
df["female"] = df["gender"].map({"female": 1, "male": 0})

# Grade: 9 = 1, 7 = 0
df["grade9"] = df["grade9"].map({1: 1, 0: 0})  # already 0/1, just re-ensuring

# Region
def classify_urbanicity(row):
    if row['living_area'] == 'Central area of the city/county':
        return 1
    elif row['living_area'] in ['Outskirts of the city/county', 
                                'The “rural-urban continuum” area of the city/county',
                                'Towns outside the city/county']:
        return 2
    elif row['living_area'] == 'Rural area':
        return 3
    else:
        return pd.NA  # or np.nan

df['urbanicity_level'] = df.apply(classify_urbanicity, axis=1)

# Siblings: 0 = 0, 1+ = 1
df["has_siblings"] = df["sibling_num"].apply(lambda x: 0 if x == 0 else 1)

# Living pattern: only "mother, father" and "mother, father, grandparents" = 1, else 0
def pattern_group(col_list):
    has_mother = "sr_w_m" in col_list
    has_father = "sr_w_f" in col_list
    has_grand = "sr_w_grand" in col_list
    has_sib = "sr_w_sib" in col_list
    has_othre = "sr_w_othre" in col_list
    has_othnon = "sr_w_othnon" in col_list

    # Case 1: mother + father only
    if has_mother and has_father and not (has_grand or has_sib or has_othre or has_othnon):
        return 1
    # Case 2: mother + father + grandparents only
    if has_mother and has_father and has_grand and not (has_sib or has_othre or has_othnon):
        return 1
    return 0

living_cols = ["sr_w_m", "sr_w_f", "sr_w_sib", "sr_w_grand", "sr_w_othre", "sr_w_othnon"]

df["living_list"] = df.apply(lambda row: [col for col in living_cols if row[col] == 1], axis=1)
df["standard_living"] = df["living_list"].apply(pattern_group)

# Check output
print(df["standard_living"].value_counts(normalize=True))

# The first time
# Select final columns
# final_df = df[[
#     "frequency_num", "parent_edu", "female", "grade9", "urban", "has_siblings", "standard_living"
# ]].dropna()

# The second time
'''
final_df = df[[
    "frequency_num", "parent_edu", "female", "grade9", "urbanicity_level", "has_siblings", "standard_living"
]].dropna()
'''
# Save for reference
df.to_csv("ceps_ols_ready.csv", index=False)

# Print sample
print(df.head())
