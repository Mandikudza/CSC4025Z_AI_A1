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

try:
    import pyagrum as gum
    import pyagrum.lib.notebook as gnb
except Exception as e:
    print("pyAgrum not available. Please install pyAgrum (pip install pyAgrum) and try again.")
    raise


class TriageInfluenceDiagram:
    """Builds an influence diagram representing triage decision.

    The class encapsulates node creation, CPT/utility definition and
    a method to compute recommended action given evidence.
    """

    def __init__(self):
        self.id = gum.InfluenceDiagram("TriageID")
        self._create_variables()
        self._add_arcs()
        self._set_cpds()
        self._set_utilities()

    def _create_variables(self):
        # Labels and number of states for each variable
        # For LabelizedVariable: arguments (name, description, numberOfStates)
        self.vars = {}
        self.vars['AgeGroup'] = self.id.add(gum.LabelizedVariable('AgeGroup', '0:Young,1:Adult,2:Elderly', 3))
        self.vars['PreExistingConditions'] = self.id.add(gum.LabelizedVariable('PreExistingConditions', '0:None,1:Moderate,2:Severe', 3))
        self.vars['SymptomSeverity'] = self.id.add(gum.LabelizedVariable('SymptomSeverity', '0:Mild,1:Moderate,2:Severe', 3))
        self.vars['OxygenSaturation'] = self.id.add(gum.LabelizedVariable('OxygenSaturation', '0:Normal(>=95),1:Low(90-94),2:Critical(<90)', 3))
        self.vars['VitalSigns'] = self.id.add(gum.LabelizedVariable('VitalSigns', '0:Stable,1:Abnormal,2:Critical', 3))
        self.vars['TestResults'] = self.id.add(gum.LabelizedVariable('TestResults', '0:Negative/Normal,1:Indicative,2:Critical', 3))
        self.vars['TrendInVitals'] = self.id.add(gum.LabelizedVariable('TrendInVitals', '0:Improving,1:Stable,2:Worsening', 3))

        # Intermediate / hidden nodes
        self.vars['Severity'] = self.id.add(gum.LabelizedVariable('Severity', '0:Low,1:Medium,2:High', 3))
        self.vars['RiskOfDeterioration'] = self.id.add(gum.LabelizedVariable('RiskOfDeterioration', '0:Low,1:Medium,2:High', 3))

        # Decision node: Action (InfluenceDiagram-specific)
        # Add decision node (LabelizedVariable) then call addDecisionNode
        self.vars['Action'] = self.id.add(gum.LabelizedVariable('Action', '0:Discharge,1:Observe,2:Ward,3:ICU', 4))
        self.id.addDecisionNode(self.vars['Action'])

        # Utility nodes (we add as numeric nodes later)
        self.vars['PatientUtility'] = self.id.add(gum.LabelizedVariable('PatientUtility', 'utility', 1))
        self.vars['ResourceCost'] = self.id.add(gum.LabelizedVariable('ResourceCost', 'cost', 1))
        self.id.addUtilityNode(self.vars['PatientUtility'])
        self.id.addUtilityNode(self.vars['ResourceCost'])

    def _add_arcs(self):
        # Connect observable features to Severity
        self.id.addArc(self.vars['AgeGroup'], self.vars['Severity'])
        self.id.addArc(self.vars['PreExistingConditions'], self.vars['Severity'])
        self.id.addArc(self.vars['SymptomSeverity'], self.vars['Severity'])
        self.id.addArc(self.vars['OxygenSaturation'], self.vars['Severity'])
        self.id.addArc(self.vars['VitalSigns'], self.vars['Severity'])
        self.id.addArc(self.vars['TestResults'], self.vars['Severity'])

        # Trend influences RiskOfDeterioration (and Severity contributes too)
        self.id.addArc(self.vars['TrendInVitals'], self.vars['RiskOfDeterioration'])
        self.id.addArc(self.vars['Severity'], self.vars['RiskOfDeterioration'])

        # Decision Action depends on Severity, ICU/Ward capacity (we will create capacity nodes as chance nodes inside CPTs or external evidence)
        # Here we add arcs from Severity and Risk to Action (decision parents)
        self.id.addArc(self.vars['Severity'], self.vars['Action'])
        self.id.addArc(self.vars['RiskOfDeterioration'], self.vars['Action'])

        # Utilities depend on Action and RiskOfDeterioration (and Severity indirectly)
        self.id.addArc(self.vars['Action'], self.vars['PatientUtility'])
        self.id.addArc(self.vars['RiskOfDeterioration'], self.vars['PatientUtility'])

        self.id.addArc(self.vars['Action'], self.vars['ResourceCost'])

    def _set_cpds(self):
        # NOTE: All CPTs below are illustrative placeholders.
        # You should replace them with values based on literature or expert elicitation.

        # For simple priors use uniform or reasonable distributions
        self.id.cpt(self.vars['AgeGroup']).fillWith([0.3, 0.5, 0.2])
        self.id.cpt(self.vars['PreExistingConditions']).fillWith([0.6, 0.3, 0.1])
        self.id.cpt(self.vars['SymptomSeverity']).fillWith([0.5, 0.3, 0.2])
        self.id.cpt(self.vars['OxygenSaturation']).fillWith([0.7, 0.2, 0.1])
        self.id.cpt(self.vars['VitalSigns']).fillWith([0.7, 0.2, 0.1])
        self.id.cpt(self.vars['TestResults']).fillWith([0.8, 0.15, 0.05])
        self.id.cpt(self.vars['TrendInVitals']).fillWith([0.2, 0.6, 0.2])

        # Severity CPT: parents: AgeGroup, PreExistingConditions, SymptomSeverity, OxygenSaturation, VitalSigns, TestResults
        # A compact but interpretable approach is to compute Severity by a simple scoring rule then convert to probabilities.
        # Since pyAgrum requires full enumerated CPT, we will create a simple rule-based CPT programmatically.
        self._fill_severity_cpt()

        # RiskOfDeterioration CPT: parents: TrendInVitals, Severity
        # We'll fill with a small table
        cpt_risk = self.id.cpt(self.vars['RiskOfDeterioration'])
        # iterate over all parent assignments
        for s in range(3):  # Severity states
            for t in range(3):  # TrendInVitals states
                # assign probabilities (Low, Medium, High)
                if s == 2 or t == 2:  # high severity or worsening trend
                    probs = [0.1, 0.2, 0.7]
                elif s == 1 or t == 1:
                    probs = [0.3, 0.5, 0.2]
                else:
                    probs = [0.8, 0.15, 0.05]
                cpt_risk[s, t, :] = probs

    def _fill_severity_cpt(self):
        # Parents order in the CPT is the order in which parents were added.
        # We need to find the parent's order for Severity node.
        cpt = self.id.cpt(self.vars['Severity'])
        parentNames = [self.id.variable(p).name() for p in cpt.variables() if p != self.vars['Severity']]
        # The above line may not return names in all pyAgrum versions; instead we will enumerate using indexes
        # We'll assume parents in insertion order: AgeGroup, PreExistingConditions, SymptomSeverity, OxygenSaturation, VitalSigns, TestResults

        # For every combination of parent states assign a distribution for Severity
        for ag in range(3):
            for pc in range(3):
                for ss in range(3):
                    for ox in range(3):
                        for vs in range(3):
                            for tr in range(3):
                                # simple scoring rule
                                score = 0
                                score += ag * 0.5
                                score += pc * 1.0
                                score += ss * 1.5
                                score += (2-ox) * 1.2  # low O2 (higher index) -> higher severity
                                score += vs * 1.3
                                score += tr * 0.8
                                # map score to probabilities (Low, Medium, High)
                                if score < 2.5:
                                    probs = [0.85, 0.13, 0.02]
                                elif score < 5.0:
                                    probs = [0.2, 0.6, 0.2]
                                else:
                                    probs = [0.05, 0.25, 0.7]
                                cpt[ag, pc, ss, ox, vs, tr, :] = probs

    def _set_utilities(self):
        # Utility tables: create tables for PatientUtility and ResourceCost
        # PatientUtility depends on Action and RiskOfDeterioration
        utilPU = gum.Potential().add(self.vars['Action']).add(self.vars['RiskOfDeterioration']).fillWith(0.0)

        # Assign utilities: higher is better (survival with no complications highest)
        # For each action (0:Discharge,1:Observe,2:Ward,3:ICU)
        # For each risk (0:Low,1:Medium,2:High)
        for a in range(4):
            for r in range(3):
                # basic heuristics: ICU best for high risk, Ward good for medium risk, Discharge ok for low risk
                if r == 2:  # High risk
                    if a == 3:  # ICU
                        u = 90
                    elif a == 2:  # Ward
                        u = 50
                    elif a == 1:  # Observe
                        u = 20
                    else:  # Discharge
                        u = -100  # bad
                elif r == 1:  # Medium risk
                    if a == 3:
                        u = 80
                    elif a == 2:
                        u = 70
                    elif a == 1:
                        u = 40
                    else:
                        u = -10
                else:  # Low risk
                    if a == 0:
                        u = 80
                    elif a == 1:
                        u = 60
                    elif a == 2:
                        u = 40
                    else:
                        u = 30
                utilPU[a, r] = u

        # ResourceCost utility: lower is better, but we'll treat as negative utility (cost)
        utilRC = gum.Potential().add(self.vars['Action']).fillWith(0.0)
        # costs: Discharge=0, Observe=10, Ward=50, ICU=200
        costs = [0, 10, 50, 200]
        for a in range(4):
            utilRC[a] = -costs[a]

        # Add utilities to the influence diagram
        self.id.utility(self.vars['PatientUtility']).copy(utilPU)
        self.id.utility(self.vars['ResourceCost']).copy(utilRC)

    def recommend_action(self, evidence: dict):
        """Given evidence (dict of variableName -> stateIndex), compute posterior and recommend an action.

        Returns a tuple (best_action_index, action_expectations) where action_expectations is a list of (action, expected_utility).
        """
        ie = gum.ShaferShenoyLIMIDInference(self.id)
        # set evidence
        for var, state in evidence.items():
            # var -> index
            if var not in self.vars:
                raise KeyError(f"Unknown variable {var}")
            ie.setEvidence({self.vars[var]: state})

        # compute expected utility for each action
        actions = list(range(4))
        action_eus = []

        for a in actions:
            # set decision to action a
            ie.addDecision({self.vars['Action']: a})
            # propagate
            ie.makeInference()
            # expected utility: sum of utilities (patient + resource)
            eu_patient = ie.expectedUtility(self.vars['PatientUtility'])
            eu_resource = ie.expectedUtility(self.vars['ResourceCost'])
            total_eu = eu_patient + eu_resource
            action_eus.append((a, total_eu))
            ie.eraseDecision()  # remove the temporary decision

        # choose best action
        best = max(action_eus, key=lambda x: x[1])
        return best[0], action_eus


def interactive_cli():
    print(textwrap.dedent("""
    Hospital Triage CLI
    Provide observed values for the following variables using the integer index shown.
    AgeGroup: 0=Young,1=Adult,2=Elderly
    PreExistingConditions: 0=None,1=Moderate,2=Severe
    SymptomSeverity: 0=Mild,1=Moderate,2=Severe
    OxygenSaturation: 0=Normal,1=Low,2=Critical
    VitalSigns: 0=Stable,1=Abnormal,2=Critical
    TestResults: 0=Normal,1=Indicative,2=Critical
    TrendInVitals: 0=Improving,1=Stable,2=Worsening
    """))

    inputs = {}
    prompts = [
        ('AgeGroup','AgeGroup (0/1/2): '),
        ('PreExistingConditions','PreExistingConditions (0/1/2): '),
        ('SymptomSeverity','SymptomSeverity (0/1/2): '),
        ('OxygenSaturation','OxygenSaturation (0/1/2): '),
        ('VitalSigns','VitalSigns (0/1/2): '),
        ('TestResults','TestResults (0/1/2): '),
        ('TrendInVitals','TrendInVitals (0/1/2): '),
    ]

    for var, prompt in prompts:
        while True:
            try:
                val = int(input(prompt))
                if val not in (0,1,2):
                    raise ValueError
                inputs[var] = val
                break
            except ValueError:
                print('Please enter 0, 1 or 2')

    triage = TriageInfluenceDiagram()

    best_action, action_eus = triage.recommend_action(inputs)

    action_names = ['Discharge','Observe','Ward','ICU']

    print('\nExpected utilities per action:')
    for a, eu in action_eus:
        print(f"  {action_names[a]}: {eu:.2f}")

    print(f"\nRecommended action: {action_names[best_action]}")


if __name__ == '__main__':
    interactive_cli()
