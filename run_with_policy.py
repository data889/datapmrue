
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Paths (edit policy_path to your file)
res_path = "resilience_panel_2016_2023.csv"
policy_path = "policy_coding.csv"  # <-- replace with your file path or use policy_coding_template.csv as a base

# Load data
res = pd.read_csv(res_path)
pol = pd.read_csv(policy_path)

# Merge
df = res.merge(pol, on="cid", how="inner")

# Post indicator (shock ~2019, use 2020+ as post)
df["post"] = (df["year"] >= 2020).astype(int)

# Baseline DID: y ~ α_i + τ_t + β (post × ΔPolicy)
# Build FE dummies
city_d = pd.get_dummies(df["cid"], prefix="cid", drop_first=True)
year_d = pd.get_dummies(df["year"], prefix="yr", drop_first=True)
X = pd.concat([df["post"]*df["DeltaPolicy_std"], city_d, year_d], axis=1)
X = sm.add_constant(X)
y = df["resilience_index"]
model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df["cid"]})
beta = model.params["post"]
se = model.bse["post"]

print("Tab.1 Baseline DID: coef(post×ΔPolicy) = %.3f, SE=%.3f, p=%.3f" % (beta, se, model.pvalues["post"]))

# Event study: interact ΔPolicy with year relative dummies k=-3..4 (relative to 2019)
df["k"] = df["year"] - 2019
ks = sorted([k for k in df["k"].unique() if -3 <= k <= 4])
for k in ks:
    df["Dk_%+d" % k] = (df["k"]==k).astype(int) * df["DeltaPolicy_std"]
# Reference period k=-1
X_es = pd.concat([df[[c for c in df.columns if c.startswith("Dk_") and c!="Dk_-1"]], city_d, year_d], axis=1)
X_es = sm.add_constant(X_es)
m_es = sm.OLS(y, X_es).fit(cov_type='cluster', cov_kwds={'groups': df["cid"]})
print("Tab.2 Event study (relative to k=-1):")
for k in ks:
    if k == -1: continue
    cname = "Dk_%+d" % k
    print("  k=%+d: coef=%.3f, SE=%.3f, p=%.3f" % (k, m_es.params[cname], m_es.bse[cname], m_es.pvalues[cname]))

# Complementarity (A×T), cross-sectional Δ levels × post
if {"DeltaA_std","DeltaT_std"} <= set(df.columns):
    df["AT"] = df["DeltaA_std"] * df["DeltaT_std"]
    X_at = pd.concat([df["post"]*df[["DeltaA_std","DeltaT_std","AT"]], city_d, year_d], axis=1)
    X_at = sm.add_constant(X_at)
    m_at = sm.OLS(y, X_at).fit(cov_type='cluster', cov_kwds={'groups': df["cid"]})
    print("Tab.8 Complementarity (post×ΔA, post×ΔT, post×ΔA×ΔT):")
    for v in ["DeltaA_std","DeltaT_std","AT"]:
        nm = "post"
        # statsmodels assigned columns may be named differently if using pd multiplication; handle safely
    for col in X_at.columns:
        if col == "const" or col.startswith("cid_") or col.startswith("yr_"): continue
        print("  %s: coef=%.3f, SE=%.3f, p=%.3f" % (col, m_at.params[col], m_at.bse[col], m_at.pvalues[col]))
else:
    print("Complementarity requires DeltaA_std and DeltaT_std in policy file.")

# Save model summaries (plain text)
with open("model_baseline_DID.txt","w",encoding="utf-8") as f:
    f.write(model.summary().as_text())

with open("model_event_study.txt","w",encoding="utf-8") as f:
    f.write(m_es.summary().as_text())

print("Done. Outputs saved: model_baseline_DID.txt, model_event_study.txt")
