import pandas as pd
import numpy as np

df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9_]', '', regex=True)

# --- AGEGROUP ---
df['AgeGroup'] = pd.cut(df['age'], bins=[0,30,65,200], labels=[0,1,2])

# --- PREEXISTING CONDITIONS ---
def preexisting(row):
    score = 0
    if row.get('creatinine_last', 0) > 1.2: score += 1
    if row.get('bun_last', 0) > 18: score += 1
    if row.get('glucose_last', 0) > 180: score += 1
    if row.get('wbc_last',0) < 4 or row.get('wbc_last',0) > 11: score += 1
    if row.get('hemoglobin_last',0) < 12: score += 1
    if row.get('platelets_last',0) < 150: score += 1
    if row.get('troponint_last',0) > 0.03: score += 1
    if row.get('lactatepoc_last',0) > 2: score += 1

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
        return 1
df['SymptomSeverity'] = df['esi'].apply(symptom_severity)

# --- OXYGEN SATURATION ---
def o2_category(o2):
    try:
        if pd.isna(o2): return 1
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
    if row.get('troponint_last',0) > 0.03 or row.get('lactatepoc_last',0) > 2: critical += 1
    if row.get('wbc_last',0) < 4 or row.get('wbc_last',0) > 12: mild += 1
    if row.get('hemoglobin_last',0) < 12: mild += 1
    if critical > 0: return 2
    elif mild > 0: return 1
    else: return 0
df['TestResults'] = df.apply(test_results, axis=1)

# --- TREND IN VITALS ---
# Try to infer change from median/last if available
def compute_trend(df):
    vitals = {
        'hr': ('triage_vital_hr', 'pulse_median'),
        'rr': ('triage_vital_rr', 'resp_median'),
        'sbp': ('triage_vital_sbp', 'sbp_median'),
        'dbp': ('triage_vital_dbp', 'dbp_median'),
        'spo2': ('triage_vital_o2', 'spo2_median'),
        'temp': ('triage_vital_temp', 'temp_median')
    }

    thresholds = {'hr':15, 'rr':5, 'sbp':15, 'dbp':10, 'spo2':3, 'temp':1.0}

    # default stable
    trend = pd.Series(1, index=df.index, dtype=int)
    worse = pd.Series(0, index=df.index, dtype=int)
    better = pd.Series(0, index=df.index, dtype=int)

    for vital, (cur, base) in vitals.items():
        if cur in df.columns and base in df.columns:
            delta = df[cur] - df[base]
            th = thresholds[vital]
            if vital in ['sbp','dbp','spo2']:
                worse += (delta < -th).astype(int)
                better += (delta > th).astype(int)
            else:
                worse += (delta > th).astype(int)
                better += (delta < -th).astype(int)

    # Assign categories
    trend.loc[worse >= 2] = 2
    trend.loc[better >= 2] = 0

    # Fallback if no baseline data
    if all(base not in df.columns for _, base in vitals.values()):
        print("⚠️ No baseline medians found; using fallback rule.")
        if 'VitalSigns' in df.columns:
            trend = df['VitalSigns'].map({2:2, 1:1, 0:0}).fillna(1).astype(int)
    return trend

df['TrendInVitals'] = compute_trend(df)

# --- Save cleaned file ---
cols_to_keep = ['AgeGroup','PreExistingConditions','SymptomSeverity','OxygenSaturation','VitalSigns','TestResults','TrendInVitals']
df_model = df[cols_to_keep]
df_model.to_csv("triage_clean_updated2.csv", index=False)

print("✅ Preprocessing complete. Saved as triage_clean_updated2.csv")
print(df_model.describe(include='all'))
