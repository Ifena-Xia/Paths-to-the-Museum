import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

d = json.load(open('ceps_final_translated.json'))
df = pd.DataFrame(d)

# drop "Other" living area -> 9761
df = df[df['living_area'] != 'Other'].copy()

freq_map = {'Never':1,'Once a year':2,'Once every half year':3,'Once a month':4,'Once a week':5,'More than once a week':6}
edu_years = {'None':0,'Finished elementary school':6,'Junior high school':9,
             'Technical secondary school or technical school':12,'Vocational high school':12,
             'Senior high school':12,'Junior college':15,'Bachelor degree':16,'Master degree or higher':19}
urb_map = {'Central area of the city/county':'urban',
           'Outskirts of the city/county':'semi',
           'The \u201crural-urban continuum\u201d area of the city/county':'semi',
           'Towns outside the city/county':'semi',
           'Rural area':'rural'}

df['frequency_n'] = df['frequency'].map(freq_map)
df['mo_y'] = df['mo_edu'].map(edu_years); df['fa_y'] = df['fa_edu'].map(edu_years)
df['parent_edu'] = df[['mo_y','fa_y']].max(axis=1)
df['urb'] = df['living_area'].map(urb_map)
df['semi_urban'] = (df['urb']=='semi').astype(int)
df['rural'] = (df['urb']=='rural').astype(int)
df['urban_core'] = (df['urb']=='urban').astype(int)
df['female'] = (df['gender']=='female').astype(int)
df['grade9'] = df['grade9'].astype(int)
df['has_siblings'] = (df['sibling_num']>=1).astype(int)
df['standard_living'] = ((df['pr_w_m']==1)&(df['pr_w_f']==1)).astype(int)
# coarse education
def edu3(y):
    return 'low' if y<=9 else ('mid' if y==12 else 'high')
df['edu3'] = df['parent_edu'].apply(edu3)

print("N =", len(df))
print("urbanicity %:", (df['urb'].value_counts(normalize=True)*100).round(1).to_dict())
print("female %%: %.1f"%(df['female'].mean()*100), " grade9 %%: %.1f"%(df['grade9'].mean()*100))
print("has_siblings %%: %.1f"%(df['has_siblings'].mean()*100), " standard_living %%: %.1f"%(df['standard_living'].mean()*100))
print("parent_edu mean %.2f sd %.2f"%(df['parent_edu'].mean(), df['parent_edu'].std()))
print("edu3 %:", (df['edu3'].value_counts(normalize=True)*100).round(1).to_dict())
print()

# ---- OLS Model 1 (no interaction) ----
m1 = smf.ols("frequency_n ~ parent_edu + semi_urban + rural + female + grade9 + has_siblings + standard_living", df).fit()
# ---- OLS Model 2 (fine edu x urbanicity interaction) ----
m2 = smf.ols("frequency_n ~ parent_edu*semi_urban + parent_edu*rural + female + grade9 + has_siblings + standard_living", df).fit()
print("=== OLS Model 1 coefficients (article: const 1.784, edu 0.0578, semi -0.1233, rural -0.4987) ===")
print(m1.params.round(4).to_dict())
print("\n=== OLS Model 2 coefficients (article: const 2.006, edu 0.0401, edu:semi 0.0293, edu:rural 0.0391) ===")
print(m2.params.round(4).to_dict())
ftest = m1.compare_f_test(m2)  # note: m1 nested in m2
print("\nF-test M1 vs M2 (fine-edu interaction): F=%.3f, p=%.5f  (article p=0.0006)"%(ftest[0],ftest[1]))

# ---- ROBUSTNESS: does the interaction survive at FINE vs COARSE education, clean data, OLS ----
m2_coarse_no = smf.ols("frequency_n ~ C(edu3) + semi_urban + rural + female + grade9 + has_siblings + standard_living", df).fit()
m2_coarse_int = smf.ols("frequency_n ~ C(edu3)*semi_urban + C(edu3)*rural + female + grade9 + has_siblings + standard_living", df).fit()
fc = m2_coarse_no.compare_f_test(m2_coarse_int)
print("\n=== ROBUSTNESS TEST (the Section 8 question) ===")
print("OLS, FINE education x urbanicity interaction:  F=%.3f p=%.5f  -> %s"%(ftest[0],ftest[1],'SIGNIFICANT' if ftest[1]<0.05 else 'n.s.'))
print("OLS, COARSE(3-cat) education x urbanicity:     F=%.3f p=%.5f  -> %s"%(fc[0],fc[1],'SIGNIFICANT' if fc[1]<0.05 else 'n.s.'))

# ---- Binary logit replicate ----
df['high_part'] = (df['frequency_n']>=3).astype(int)
print("\nbinary high_part share (0 = never/once-a-year):", round((1-df['high_part'].mean())*100,1),"% are 0 (article 67.5)")
b_no = smf.logit("high_part ~ C(edu3) + semi_urban + rural + female + grade9 + has_siblings + standard_living", df).fit(disp=0)
b_int = smf.logit("high_part ~ C(edu3)*semi_urban + C(edu3)*rural + female + grade9 + has_siblings + standard_living", df).fit(disp=0)
print("Binary logit COARSE edu: BIC no-interaction=%.2f  with-interaction=%.2f (article 11574.48 vs 11603.5)"%(b_no.bic,b_int.bic))
# fine-edu interaction in logit
b_fine_no = smf.logit("high_part ~ parent_edu + semi_urban + rural + female + grade9 + has_siblings + standard_living", df).fit(disp=0)
b_fine_int = smf.logit("high_part ~ parent_edu*semi_urban + parent_edu*rural + female + grade9 + has_siblings + standard_living", df).fit(disp=0)
lr = 2*(b_fine_int.llf - b_fine_no.llf)
from scipy.stats import chi2
p_lr = chi2.sf(lr, 2)
print("Binary logit FINE edu x urbanicity: LR chi2(2)=%.3f p=%.5f -> %s; BIC no=%.2f with=%.2f"%(lr,p_lr,'SIGNIFICANT' if p_lr<0.05 else 'n.s.',b_fine_no.bic,b_fine_int.bic))

print("\n\n================ FIXED p-values + standard_living check ================")
from scipy.stats import f as fdist
# fine interaction: 2 added params
F_fine, df_fine = 7.073, (2, m2.df_resid)
p_fine = fdist.sf(7.073, 2, m2.df_resid)
F_coarse = 1.849
p_coarse = fdist.sf(1.849, 4, m2_coarse_int.df_resid)
print("OLS FINE edu interaction:   F=7.073 df=(2,%d)  p=%.5f"%(m2.df_resid,p_fine))
print("OLS COARSE edu interaction: F=1.849 df=(4,%d)  p=%.5f"%(m2_coarse_int.df_resid,p_coarse))

# standard_living candidates
cand = {
 'pr both parents': ((df['pr_w_m']==1)&(df['pr_w_f']==1)).mean(),
 'sr both parents': ((df['sr_w_m']==1)&(df['sr_w_f']==1)).mean(),
 'pr both, no other rel/nonrel': ((df['pr_w_m']==1)&(df['pr_w_f']==1)&(df['pr_w_othre']==0)&(df['pr_w_othnon']==0)).mean(),
 'sr both, no other rel/nonrel': ((df['sr_w_m']==1)&(df['sr_w_f']==1)&(df['sr_w_othre']==0)&(df['sr_w_othnon']==0)).mean(),
 'pr both & no sibling co-res': ((df['pr_w_m']==1)&(df['pr_w_f']==1)&(df['pr_w_sib']==0)).mean(),
 'relationshi biological mother': (df['relationshi']=='Biological mother').mean(),
}
print("\nstandard_living candidate shares (target ~48%):")
for k,v in cand.items(): print(f"  {v*100:5.1f}%  {k}")
