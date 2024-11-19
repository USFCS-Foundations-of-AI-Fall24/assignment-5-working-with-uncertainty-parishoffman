from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ],
    evidence=["Gas", "Ignition", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts": ['yes', 'no'], "Gas": ['Full', "Empty"], "Ignition": ["Works", "Doesn't work"], "KeyPresent": ["yes", "no"],},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

cpd_key_present = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]},
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))


def main() :
    print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works"}))
    radio_prob_without_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    radio_prob_with_gas = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print("Without evidence of gas:", radio_prob_without_gas)
    print("With evidence of gas:", radio_prob_with_gas)
    ignition_prob_without_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    ignition_prob_with_no_gas = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print("Without evidence of gas:", ignition_prob_without_gas)
    print("With evidence of no gas:", ignition_prob_with_no_gas)
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))
    print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))

main()