import pandas as pd

# Load your dataset
df = pd.read_json("ceps_final_translated.json")

# Define categories
low_participation = ["Never", "Once a year"]
moderate_participation = ["Once every half year"]

# Recode frequency into three levels
def recode_freq(freq):
    if freq in low_participation:
        return "low"
    elif freq in moderate_participation:
        return "moderate"
    else:
        return "high"

df["participation_level"] = df["frequency"].apply(recode_freq)

# Define education levels
low_edu = ["None", "Finished elementary school", "Junior high school"]
med_edu = ["Technical secondary school or technical school", "Vocational high school", "Senior high school"]
high_edu = ["Junior college", "Bachelor degree", "Master degree or higher"]

# Convert to ordinal code
def edu_code(edu):
    if edu in low_edu:
        return 1
    elif edu in med_edu:
        return 2
    elif edu in high_edu:
        return 3
    else:
        return None

df["mo_code"] = df["mo_edu"].apply(edu_code)
df["fa_code"] = df["fa_edu"].apply(edu_code)

# Take the higher education level between the parents
df["parent_code"] = df[["mo_code", "fa_code"]].max(axis=1)

# Create dummy variables
df["parent_edu_l"] = (df["parent_code"] == 1).astype(int)
df["parent_edu_m"] = (df["parent_code"] == 2).astype(int)
df["parent_edu_h"] = (df["parent_code"] == 3).astype(int)

df["female"] = (df["gender"] == "female").astype(int)
df["has_siblings"] = (df["sibling_num"] > 0).astype(int)

# Define mapping function
def map_urbanicity(area):
    if area == "Central area of the city/county":
        return "urban_core"
    elif area in [
        "Outskirts of the city/county", 
        "Town outside the city/county", 
        "Rural-urban continuum area"
    ]:
        return "semi_urban"
    elif area == "Rural area":
        return "rural"
    else:
        return None  # "Other" will be excluded later

df["urbanicity_level"] = df["living_area"].apply(map_urbanicity)

# Drop ambiguous rows (urbanicity_level is None)
df = df[df["urbanicity_level"].notnull()]

# Define standard living condition
def is_standard_living(row):
    return int(
        row["pr_w_m"] == 1 and row["pr_w_f"] == 1 and (
            row["pr_w_grand"] == 0 or pd.notnull(row["pr_w_grand"])
        )
    )

df["standard_living"] = df.apply(is_standard_living, axis=1)

# Save cleaned dataset
df.to_csv("ceps_mlogit_ready.csv", index=False)
