# Hospital Triage Decision Support System

## Overview

This project implements a Bayesian decision network (influence diagram) for hospital triage using pyAgrum in Python. The model estimates patient severity and risk of deterioration based on clinical inputs and recommends actions (Discharge, Observe, Ward, ICU) by maximizing expected utility, balancing patient outcomes and resource costs.

### The system includes:

- triage_system_new.py: Main script with the influence diagram class and CLI for interactive use.

- preprocess_triage.py: Preprocessing script to transform raw ED data into categorical variables for the model.

- processed_data.csv: Sample raw dataset (anonymized ED records).

- triage_clean_updated2.csv: Sample preprocessed dataset.

## Requirements

- Python 3.6+

- Libraries: pandas, numpy, pyagrum (install with pip install pyagrum pandas numpy)

## How the Code Works

1. Preprocessing (preprocess_triage.py):

   **This step (Step 1) does not need to be done by the user as it has already been executed. It was merely mentioned for clarity on how the code works**

   - Loads raw data (e.g., ESI, age, vital signs, lab results).

   - Categorizes into model variables (0-2 scale): AgeGroup, PreExistingConditions (scored from labs), SymptomSeverity (from ESI), OxygenSaturation, VitalSigns (abnormal count), TestResults (critical/mild), TrendInVitals (fallback to VitalSigns if no medians).

   - Saves categorized data for model use.

2. Model Construction (triage_system_new.py):

   - Creates chance nodes for inputs and intermediates (Severity, RiskOfDeterioration).

   - Adds arcs for dependencies (inputs → Severity → Risk → Action).

   - Sets CPTs with placeholders (heuristic scores for Severity; logic for Risk).

   - Sets utilities for PatientUtility (QALY-based) and ResourceCost (cost-based).

   - Updates root priors from dataset if chosen.

3. Inference and Recommendation:

   - Uses ShaferShenoyLIMIDInference to compute posteriors and expected utilities for each action.

   - Recommends the action with max total EU (Patient + Resource).

4. CLI Interaction:

   - Prompts for patient inputs or loads dataset.

   - Outputs expected utilities and recommended action.

## How to Run

1. Preprocess Data (optional, if raw data available):

   ```
   python preprocess_triage.py
   ```

   - Outputs triage_clean_updated2.csv.

2. Run the Model:

   ```
   python triage_system_new.py
   ```

   - Follow CLI prompts: Choose to load dataset (enter path to CSV), input patient values (0-2).

   - Outputs recommendation.

## Example Usage

- Load priors from triage_clean_updated2.csv when prompted.

- Enter patient data, e.g., AgeGroup=2, PreExistingConditions=1, etc.

- View recommended action, e.g., "Ward".

## Limitations

- CPTs and utilities are placeholders; validate with domain experts.

- Requires pyAgrum for inference.

- Small datasets may lead to sparse priors.

For questions, contact the authors: Emma van der Berg(VBREMM005@myuct.ac.za), Maryam Mather(MTHMAR046@myuct.ac.za), Mandikudza Dangwa(DNGMAN001@myuct.ac.za)
