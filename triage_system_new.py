"""
Triage Decision System using Bayesian / Influence Networks (pyAgrum)

File: triage_system.py
Author: Mandi, Emma, Maryam

This module constructs a Bayesian decision network (influence diagram)
for a simple hospital triage system and provides a CLI for interacting
with the model.

NOTES:
- This code uses pyAgrum. Install with: pip install pyAgrum
- CPT (probability) values and utility values are placeholders and MUST
  be adjusted using domain knowledge, literature, or expert elicitation.
- The model contains chance nodes: AgeGroup, PreExistingConditions,
  SymptomSeverity, OxygenSaturation, VitalSigns, TestResults, TrendInVitals,
  Severity, RiskOfDeterioration.
  Decision node: Action (ICU, Ward, Observe, Discharge).
  Utility nodes: PatientUtility and ResourceCost.

Usage:
    python triage_system.py

The CLI will ask for observable values and then compute posterior
probabilities, expected utilities for each action, and recommend the
best action according to maximum expected utility.

"""

import sys
import textwrap
import pandas as pd

import pyagrum as gum


class TriageInfluenceDiagram:
    """Influence diagram for hospital triage decisions."""

    def __init__(self):
        self.id = gum.InfluenceDiagram()
        self.id.name = "TriageID"
        self._create_variables()
        self._add_arcs()
        self._set_cpds()
        self._set_utilities()

    def _create_variables(self):
        # Root / chance nodes
        self.vars = {}
        self.vars['AgeGroup'] = self.id.add(gum.LabelizedVariable('AgeGroup', '0:Young,1:Adult,2:Elderly', 3))
        self.vars['PreExistingConditions'] = self.id.add(gum.LabelizedVariable('PreExistingConditions', '0:None,1:Moderate,2:Severe', 3))
        self.vars['SymptomSeverity'] = self.id.add(gum.LabelizedVariable('SymptomSeverity', '0:Mild,1:Moderate,2:Severe', 3))
        self.vars['OxygenSaturation'] = self.id.add(gum.LabelizedVariable('OxygenSaturation', '0:Normal,1:Low,2:Critical', 3))
        self.vars['VitalSigns'] = self.id.add(gum.LabelizedVariable('VitalSigns', '0:Stable,1:Abnormal,2:Critical', 3))
        self.vars['TestResults'] = self.id.add(gum.LabelizedVariable('TestResults', '0:Normal,1:Indicative,2:Critical', 3))
        self.vars['TrendInVitals'] = self.id.add(gum.LabelizedVariable('TrendInVitals', '0:Improving,1:Stable,2:Worsening', 3))

        # Intermediate / hidden nodes
        self.vars['Severity'] = self.id.add(gum.LabelizedVariable('Severity', '0:Low,1:Medium,2:High', 3))
        self.vars['RiskOfDeterioration'] = self.id.add(gum.LabelizedVariable('RiskOfDeterioration', '0:Low,1:Medium,2:High', 3))

        # Decision node: Action (do NOT create LabelizedVariable separately)
        self.id.addDecisionNode('Action', 4)  # 4 states: Discharge, Observe, Ward, ICU
        self.vars['Action'] = self.id.variable('Action')  # get reference to the created node

        # Utility nodes
        self.id.addUtilityNode('PatientUtility')
        self.vars['PatientUtility'] = self.id.variable('PatientUtility')

        self.id.addUtilityNode('ResourceCost')
        self.vars['ResourceCost'] = self.id.variable('ResourceCost')


    def _add_arcs(self):
        # Observable -> Severity
        for var in ['AgeGroup', 'PreExistingConditions', 'SymptomSeverity',
                    'OxygenSaturation', 'VitalSigns', 'TestResults']:
            self.id.addArc(var, 'Severity')

        # Trend + Severity -> RiskOfDeterioration
        self.id.addArc('TrendInVitals', 'RiskOfDeterioration')
        self.id.addArc('Severity', 'RiskOfDeterioration')

        # Severity + Risk -> Action
        self.id.addArc('Severity', 'Action')

        # Utilities depend on Action + Risk
        self.id.addArc('Action', 'PatientUtility')
        self.id.addArc('RiskOfDeterioration', 'PatientUtility')
        self.id.addArc('Action', 'ResourceCost')

    def _set_cpds(self):
        # Simple root priors
        self.id.cpt('AgeGroup').fillWith([0.3, 0.5, 0.2])
        self.id.cpt('PreExistingConditions').fillWith([0.6, 0.3, 0.1])
        self.id.cpt('SymptomSeverity').fillWith([0.5, 0.3, 0.2])
        self.id.cpt('OxygenSaturation').fillWith([0.7, 0.2, 0.1])
        self.id.cpt('VitalSigns').fillWith([0.7, 0.2, 0.1])
        self.id.cpt('TestResults').fillWith([0.8, 0.15, 0.05])
        self.id.cpt('TrendInVitals').fillWith([0.2, 0.6, 0.2])

        # Severity CPT placeholder
        cpt_sev = self.id.cpt('Severity')
        for ag in range(3):
            for pc in range(3):
                for ss in range(3):
                    for ox in range(3):
                        for vs in range(3):
                            for tr in range(3):
                                score = ag*0.5 + pc*1.0 + ss*1.5 + (2-ox)*1.2 + vs*1.3 + tr*0.8
                                if score < 2.5:
                                    probs = [0.85, 0.13, 0.02]
                                elif score < 5.0:
                                    probs = [0.2, 0.6, 0.2]
                                else:
                                    probs = [0.05, 0.25, 0.7]
                                cpt_sev[ag, pc, ss, ox, vs, tr, :] = probs

        # RiskOfDeterioration CPT placeholder
        cpt_risk = self.id.cpt('RiskOfDeterioration')
        for s in range(3):
            for t in range(3):
                if s == 2 or t == 2:
                    probs = [0.1, 0.2, 0.7]
                elif s == 1 or t == 1:
                    probs = [0.3, 0.5, 0.2]
                else:
                    probs = [0.8, 0.15, 0.05]
                cpt_risk[s, t, :] = probs

    def _set_utilities(self):
        util_pu = self.id.utility('PatientUtility')
        util_rc = self.id.utility('ResourceCost')

        # Set PatientUtility values
        inst_pu = gum.Instantiation(util_pu)
        for a in range(4):  # Action: 0=Discharge, 1=Observe, 2=Ward, 3=ICU
            for r in range(3):  # Risk: 0=Low, 1=Medium, 2=High
                if r == 2:  # High risk
                    u = {3: 180, 2: 100, 1: 40, 0: -100}[a]  # ICU big gain, Discharge harmful
                elif r == 1:  # Medium risk
                    u = {3: 120, 2: 90, 1: 45, 0: -10}[a]    # Ward good, ICU strong
                else:  # Low risk
                    u = {0: 50, 1: 40, 2: 30, 3: 20}[a]      # Discharge best, ICU overkill
                inst_pu.chgVal('Action', a)
                inst_pu.chgVal('RiskOfDeterioration', r)
                util_pu.set(inst_pu, u)

        # Set ResourceCost values
        inst_rc = gum.Instantiation(util_rc)
        costs = [0, 8, 50, 110]  # Discharge=0, Observe=8, Ward=50, ICU=110
        for a in range(4):
            inst_rc.chgVal('Action', a)
            util_rc.set(inst_rc, -costs[a])
    
    
    def recommend_action(self, evidence: dict):
        """
        Compute expected utility for each action and recommend the best one.
        """
        action_eus = []
        
        for a in range(4):  # 4 actions: Discharge, Observe, Ward, ICU
            try:
                # Create a new inference engine
                ie = gum.ShaferShenoyLIMIDInference(self.id)
                
                # Add ALL evidence including the observed variables
                for var, state in evidence.items():
                    ie.addEvidence(var, state)
                
                # Set the action as evidence
                ie.addEvidence('Action', a)
                
                # Make inference
                ie.makeInference()
                
                # Get the posterior distribution for RiskOfDeterioration
                risk_posterior = ie.posterior('RiskOfDeterioration')
                print(f"Debug Action {a}: Risk posterior: {[risk_posterior[i] for i in range(3)]}")
                
                # Get expected utilities
                eu_patient_tensor = ie.posteriorUtility('PatientUtility')
                eu_resource_tensor = ie.posteriorUtility('ResourceCost')
                
                # Debug the tensor structures
                patient_structure = eu_patient_tensor.tolist()
                resource_structure = eu_resource_tensor.tolist()
                print(f"Debug Action {a}: PatientUtility structure: {patient_structure}")
                print(f"Debug Action {a}: ResourceCost structure: {resource_structure}")
                
                # Extract values - handle the nested list structure properly
                def extract_utility(tensor):
                    """Extract utility value from potentially nested tensor structure"""
                    raw = tensor.tolist()
                    
                    # The structure seems to be nested lists, we need to find the actual value
                    # Based on previous output, it might be a 3D structure for PatientUtility
                    # and 1D for ResourceCost
                    
                    if isinstance(raw, list):
                        # For PatientUtility: [[[50.0], [40.0], ...]] - we need to find the right value
                        # For a given action and risk posterior, we should compute weighted average
                        if len(raw) == 3:  # Probably risk levels
                            # This is likely the full utility table, we need to compute expected value
                            # using the risk posterior distribution
                            expected_value = 0.0
                            for risk_level in range(3):
                                risk_prob = risk_posterior[risk_level]
                                # Find the utility for this risk level and action
                                risk_utils = raw[risk_level]
                                if isinstance(risk_utils, list) and len(risk_utils) == 4:  # Actions
                                    action_util = risk_utils[a]
                                    if isinstance(action_util, list) and len(action_util) == 1:
                                        action_util = action_util[0]
                                    expected_value += risk_prob * action_util
                            return expected_value
                        elif len(raw) == 1:
                            return extract_utility(raw[0])
                        elif len(raw) == 4:  # Probably actions for ResourceCost
                            return raw[a] if not isinstance(raw[a], list) else raw[a][0]
                    return float(raw)
                
                eu_patient = extract_utility(eu_patient_tensor)
                eu_resource = extract_utility(eu_resource_tensor)
                total_eu = eu_patient + eu_resource
                
                print(f"Action {a}: Patient EU = {eu_patient:.2f}, Resource EU = {eu_resource:.2f}, Total = {total_eu:.2f}")
                
                action_eus.append((a, total_eu))
                
            except Exception as e:
                print(f"Error computing EU for action {a}: {str(e)}")
                import traceback
                traceback.print_exc()
                action_eus.append((a, -999.0))
        
        # Find best action
        if action_eus:
            best = max(action_eus, key=lambda x: x[1])
            return best[0], action_eus
        else:
            return 0, [(0, 0), (1, 0), (2, 0), (3, 0)]
    
    '''
    def recommend_action(self, evidence: dict):
        """
        Compute expected utility for each action and recommend the best one.
        """
        # For each action, compute the expected utility
        action_eus = []
        for a in range(4):  # 4 actions: Discharge, Observe, Ward, ICU
            # Create a new inference engine for this specific action scenario
            ie_action = gum.ShaferShenoyLIMIDInference(self.id)
            
            # Add all evidence
            for var, state in evidence.items():
                ie_action.addEvidence(var, state)
            
            # Add the decision as evidence
            ie_action.addEvidence('Action', a)
            
            # Make inference
            ie_action.makeInference()
            
            # Get expected utilities - posteriorUtility returns a Tensor
            eu_patient_tensor = ie_action.posteriorUtility('PatientUtility')
            eu_resource_tensor = ie_action.posteriorUtility('ResourceCost')
            
            # Extract scalar values from tensors - handle nested lists
            patient_list = eu_patient_tensor.tolist()
            resource_list = eu_resource_tensor.tolist()
            
            # Flatten if needed and get the value
            while isinstance(patient_list, list):
                patient_list = patient_list[0]
            while isinstance(resource_list, list):
                resource_list = resource_list[0]
                
            eu_patient = float(patient_list)
            eu_resource = float(resource_list)
            total_eu = eu_patient + eu_resource
            
            action_eus.append((a, total_eu))
        
        # Find best action
        best = max(action_eus, key=lambda x: x[1])
        return best[0], action_eus
    
    '''

    def _update_priors_from_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        root_nodes = ['AgeGroup','PreExistingConditions','SymptomSeverity',
                      'OxygenSaturation','VitalSigns','TestResults','TrendInVitals']
        for var in root_nodes:
            if var not in df.columns: 
                continue
            counts = df[var].value_counts(normalize=True).sort_index()
            probs = [counts.get(i, 0.0) for i in range(3)]
            # Normalize to ensure sum is 1.0
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
            else:
                probs = [1/3, 1/3, 1/3]
            self.id.cpt(var).fillWith(probs)
            
def interactive_cli():
    print(textwrap.dedent("""
    ==========================================
    Hospital Triage Decision Support System
    ==========================================
    
    Provide observed values for the following variables using the integer index shown.
    
    AgeGroup: 0=Young, 1=Adult, 2=Elderly
    PreExistingConditions: 0=None, 1=Moderate, 2=Severe
    SymptomSeverity: 0=Mild, 1=Moderate, 2=Severe
    OxygenSaturation: 0=Normal, 1=Low, 2=Critical
    VitalSigns: 0=Stable, 1=Abnormal, 2=Critical
    TestResults: 0=Normal, 1=Indicative, 2=Critical
    TrendInVitals: 0=Improving, 1=Stable, 2=Worsening
    """))

    triage = TriageInfluenceDiagram()

    # Optional: update priors from CSV dataset
    use_data = input("Load priors from dataset? (y/n): ").strip().lower()
    if use_data == "y":
        csv_path = input("Enter path to dataset CSV: ").strip()
        try:
            triage._update_priors_from_dataset(csv_path)
            print("Priors updated from dataset")
        except Exception as e:
            print(f"Could not load dataset: {e}")
            print("Continuing with default priors...")

    # Collect user inputs
    inputs = {}
    prompts = [
        ('AgeGroup','AgeGroup (0=Young/1=Adult/2=Elderly): '),
        ('PreExistingConditions','PreExistingConditions (0=None/1=Moderate/2=Severe): '),
        ('SymptomSeverity','SymptomSeverity (0=Mild/1=Moderate/2=Severe): '),
        ('OxygenSaturation','OxygenSaturation (0=Normal/1=Low/2=Critical): '),
        ('VitalSigns','VitalSigns (0=Stable/1=Abnormal/2=Critical): '),
        ('TestResults','TestResults (0=Normal/1=Indicative/2=Critical): '),
        ('TrendInVitals','TrendInVitals (0=Improving/1=Stable/2=Worsening): '),
    ]

    print("\n--- Patient Information ---")
    for var, prompt in prompts:
        while True:
            try:
                val = int(input(prompt))
                if val not in (0,1,2):
                    raise ValueError
                inputs[var] = val
                break
            except ValueError:
                print("Please enter 0, 1, or 2.")

    # Compute recommended action
    print("\n--- Computing recommendation... ---")
    best_action, action_eus = triage.recommend_action(inputs)
    action_names = ['Discharge','Observe','Ward','ICU']

    print("\n==========================================")
    print("Expected Utilities per Action:")
    print("==========================================")
    for a, eu in action_eus:
        marker = " ← RECOMMENDED" if a == best_action else ""
        print(f"  {action_names[a]:10s}: {eu:7.2f}{marker}")

    print(f"\n✓ RECOMMENDED ACTION: {action_names[best_action]}")
    print("==========================================\n")
    
if __name__ == '__main__':
    interactive_cli()