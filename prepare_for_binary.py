import pandas as pd

# Load data
df = pd.read_json("ceps_final_translated.json")

# 1) Recode frequency into binary: 0 = Never or Once a year, 1 = all other frequencies
low_freq = ["Never", "Once a year"]
df["freq_binary"] = df["frequency"].apply(lambda x: 0 if x in low_freq else 1)

# 2) Recode parental education into 3 levels
low_edu = ["None", "Finished elementary school", "Junior high school"]
med_edu = ["Technical secondary school or technical school", "Vocational high school", "Senior high school"]
high_edu = ["Junior college", "Bachelor degree", "Master degree or higher"]

def edu_level(edu):
    if edu in low_edu:
        return "l"
    elif edu in med_edu:
        return "m"
    elif edu in high_edu:
        return "h"
    else:
        return None

df["mo_edu_level"] = df["mo_edu"].apply(edu_level)
df["fa_edu_level"] = df["fa_edu"].apply(edu_level)

# Determine higher parent's education level
def get_parent_level(row):
    levels = {"l": 0, "m": 1, "h": 2}
    mo = levels.get(row["mo_edu_level"], -1)
    fa = levels.get(row["fa_edu_level"], -1)
    if mo == -1 and fa == -1:
        return None
    elif mo >= fa:
        return row["mo_edu_level"]
    else:
        return row["fa_edu_level"]

df["parent_edu"] = df.apply(get_parent_level, axis=1)

# 3) Gender: female = 1, male = 0
df["female"] = df["gender"].map({"female": 1, "male": 0})

# 4) Grade: grade9 is already 0/1

# 5) Urbanicity levels
def classify_urbanicity(area):
    if area == "Central area of the city/county":
        return 1  # Urban core
    elif area in [
        "Outskirts of the city/county",
        "The “rural-urban continuum” area of the city/county",
        "Towns outside the city/county"
    ]:
        return 2  # Semi-urban
    elif area == "Rural area":
        return 3  # Rural
    else:
        return None  # Drop 'Other' and unrecognized

df["urbanicity_level"] = df["living_area"].apply(classify_urbanicity)

# 6) Siblings
df["has_siblings"] = df["sibling_num"].apply(lambda x: 0 if x == 0 else 1)

# 7) Household composition: standard_living
def pattern_group(col_list):
    has_mother = "sr_w_m" in col_list
    has_father = "sr_w_f" in col_list
    has_grand = "sr_w_grand" in col_list
    has_sib = "sr_w_sib" in col_list
    has_othre = "sr_w_othre" in col_list
    has_othnon = "sr_w_othnon" in col_list

    if has_mother and has_father and not (has_grand or has_sib or has_othre or has_othnon):
        return 1
    if has_mother and has_father and has_grand and not (has_sib or has_othre or has_othnon):
        return 1
    return 0

living_cols = ["sr_w_m", "sr_w_f", "sr_w_sib", "sr_w_grand", "sr_w_othre", "sr_w_othnon"]
df["living_list"] = df.apply(lambda row: [col for col in living_cols if row[col] == 1], axis=1)
df["standard_living"] = df["living_list"].apply(pattern_group)

# Filter usable data
df_clean = df.dropna(subset=[
    "freq_binary", "parent_edu", "female", "grade9",
    "urbanicity_level", "has_siblings", "standard_living"
])

# Optional: Save or preview
df_clean.to_csv("ceps_binary_ready.csv", index=False)
print(df_clean.head())
