import pandas as pd
import numpy as np

df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9_]', '', regex=True)

# --- AGEGROUP ---
df['AgeGroup'] = pd.cut(df['age'], bins=[0,30,65,200], labels=[0,1,2])

# --- PREEXISTING CONDITIONS ---
# Use more labs + broaden thresholds
def preexisting(row):
    score = 0
    # Kidney/liver markers
    if row.get('creatinine_last', 0) > 1.2: score += 1
    if row.get('bun_last', 0) > 18: score += 1
    if row.get('glucose_last', 0) > 180: score += 1
    # Hematology
    if row.get('wbc_last',0) < 4 or row.get('wbc_last',0) > 11: score += 1
    if row.get('hemoglobin_last',0) < 12: score += 1
    if row.get('platelets_last',0) < 150: score += 1
    # Troponin/lactate
    if row.get('troponint_last',0) > 0.03: score += 1
    if row.get('lactatepoc_last',0) > 2: score += 1

    # Map score to 0/1/2
    if score == 0: return 0
    elif score <= 3: return 1
    else: return 2

df['PreExistingConditions'] = df.apply(preexisting, axis=1)

# --- SYMPTOM SEVERITY ---
def symptom_severity(esi):
    try:
        esi = int(esi)
        if esi <= 2: return 2
        elif esi == 3: return 1
        else: return 0
    except:
        return 1  # default to moderate if missing
df['SymptomSeverity'] = df['esi'].apply(symptom_severity)

# --- OXYGEN SATURATION ---
def o2_category(o2):
    try:
        if pd.isna(o2): return 1  # default to low-normal
        o2 = float(o2)
        if o2 >= 95: return 0
        elif o2 >= 90: return 1
        else: return 2
    except:
        return 1
df['OxygenSaturation'] = df['triage_vital_o2'].apply(o2_category)

# --- VITAL SIGNS ---
def vital_signs(row):
    abnormal = 0
    hr = row.get('triage_vital_hr', np.nan)
    rr = row.get('triage_vital_rr', np.nan)
    sbp = row.get('triage_vital_sbp', np.nan)
    dbp = row.get('triage_vital_dbp', np.nan)
    temp = row.get('triage_vital_temp', np.nan)

    if pd.notna(hr) and (hr < 55 or hr > 105): abnormal += 1
    if pd.notna(rr) and (rr < 12 or rr > 22): abnormal += 1
    if pd.notna(sbp) and sbp < 95: abnormal += 1
    if pd.notna(dbp) and dbp < 60: abnormal += 1
    if pd.notna(temp) and (temp < 36 or temp > 38): abnormal += 1

    if abnormal == 0: return 0
    elif abnormal <= 2: return 1
    else: return 2
df['VitalSigns'] = df.apply(vital_signs, axis=1)

# --- TEST RESULTS ---
def test_results(row):
    critical = 0
    mild = 0
    # Troponin/Lactate = critical
    if row.get('troponint_last',0) > 0.03 or row.get('lactatepoc_last',0) > 2: critical += 1
    # WBC/Hgb = mild
    if row.get('wbc_last',0) < 4 or row.get('wbc_last',0) > 12: mild += 1
    if row.get('hemoglobin_last',0) < 12: mild += 1
    if critical > 0: return 2
    elif mild > 0: return 1
    else: return 0
df['TestResults'] = df.apply(test_results, axis=1)

# --- TREND IN VITALS ---
df['TrendInVitals'] = 1  # stable

# --- Save cleaned file ---
cols_to_keep = ['AgeGroup','PreExistingConditions','SymptomSeverity','OxygenSaturation','VitalSigns','TestResults','TrendInVitals']
df_model = df[cols_to_keep]
df_model.to_csv("triage_clean_updated.csv", index=False)

print("Preprocessing complete, saved as triage_clean_updated.csv")
print(df_model.describe())
