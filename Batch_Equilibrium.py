import numpy as np
import pandas as pd
from tc_python import TCPython, ThermodynamicQuantity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_lattice_parameter(molar_volume):
    avogadro_number = 6.023e23
    coordination_number = 4
    lattice_parameter = ((molar_volume * coordination_number) / avogadro_number) ** (1 / 3)
    return lattice_parameter * 1e10  # Convert from meters to angstroms

def diffusivity(D_0, Q, temperature):
    R = 8.314  # Universal gas constant in J/(mol K)
    D_0 = float(D_0)
    Q = float(Q)
    D = (D_0 * np.exp(-Q / (R * temperature)))  # Convert to m^2/s
    return D

def list_stable_phases_and_calculations(results, diffusivity_data, temperature):
    data = []
    stable_phases = results.get_stable_phases()
    phase_data = {}

    for phase in stable_phases:
        amount = results.get_value_of(f'NP({phase})')
        molar_volume = results.get_value_of(f'VM({phase})') * 1e6  # Convert from m^3 to cm^3
        lattice_parameter = calculate_lattice_parameter(molar_volume * 1e-6)  # Convert back to m^3 for calculation

        mole_fractions = {}
        for component in results.get_components():
            mole_fraction = results.get_value_of(f'X({phase},{component})')
            mole_fractions[component] = mole_fraction

        diffusivities = {}
        for component, values in diffusivity_data.items():
            try:
                D = diffusivity(values['D_0'], values['Q'], temperature)
                diffusivities[component] = D
            except ValueError:
                diffusivities[component] = np.nan

        data.append([phase, amount, molar_volume, lattice_parameter, mole_fractions, diffusivities])
        phase_data[phase] = {
            'amount': amount,
            'lattice_parameter': lattice_parameter,
            'mole_fractions': mole_fractions,
            'diffusivities': diffusivities
        }

    return data, phase_data

input_file = 'Final_Impurity_data.csv'
systems_file = 'trail_28_run.csv'

try:
    diffusivity_df = pd.read_csv(input_file, usecols=['component', 'D_0', 'Q'], index_col='component')
    diffusivity_df = diffusivity_df.astype({'D_0': 'float64', 'Q': 'float64'})
    systems_df = pd.read_csv(systems_file)
except FileNotFoundError as e:
    logging.error(f"Error: {e}")
    exit(1)
except ValueError as e:
    logging.error(f"Error: {e}")
    logging.error("Please ensure the input files have the correct format and columns.")
    exit(1)

diffusivity_data = diffusivity_df.to_dict(orient='index')

output_results = []

from tc_python.single_equilibrium import SingleEquilibriumOptions, SingleEquilibriumCalculation

with TCPython() as start:
    options = SingleEquilibriumOptions()
    options.set_max_no_of_iterations(50000)
    options.set_global_minimization_max_grid_points(200000)
    options.set_required_accuracy(1e-7)
    options.set_smallest_fraction(1e-12)
    options.enable_control_step_size_during_minimization()
    options.enable_force_positive_definite_phase_hessian()
    options.enable_approximate_driving_force_for_metastable_phases()

    for index, system in systems_df.iterrows():
        system_name = system['system_name']
        temperature = system['temperature']
        mass_fractions = {element: system[element] for element in systems_df.columns if not pd.isna(system[element]) and element not in ['temperature', 'system_name']}

        logging.info(f"Processing system {index + 1} with elements: {list(mass_fractions.keys())} and temperature: {temperature}")

        if not np.isclose(sum(mass_fractions.values()), 1.0, atol=1e-3):
            error_message = f"Mass fractions do not sum to 1. Skipping this system."
            logging.error(f"System {index + 1}: {error_message}")
            output_results.append({
                "System Index": index + 1,
                "System Name": system_name,
                "Temperature": temperature,
                "Error": error_message
            })
            continue

        element_to_ignore = max(mass_fractions, key=mass_fractions.get)
        logging.info(f"System {index + 1}: Ignoring element {element_to_ignore} for mass fraction condition.")

        try:
            calculation = (
                start
                .select_database_and_elements("TCNI12", list(mass_fractions.keys()))
                .get_system()
                .with_single_equilibrium_calculation()
                .set_condition(ThermodynamicQuantity.temperature(), temperature)
                .set_condition(ThermodynamicQuantity.pressure(), 101325)
            )

            for element, mass_fraction in mass_fractions.items():
                if element != element_to_ignore:
                    calculation.set_condition(ThermodynamicQuantity.mass_fraction_of_a_component(element), mass_fraction)

            calculation.enable_global_minimization()
            calculation.with_options(options)

            logging.info(f"System {index + 1}: Starting calculation")
            results = calculation.calculate()
            logging.info(f"System {index + 1}: Calculation completed")

            data, phase_data = list_stable_phases_and_calculations(results, diffusivity_data, temperature)

            fcc_l12_phases = {phase: data for phase, data in phase_data.items() if phase.startswith("FCC_L12#")}
            sorted_fcc_l12_phases = sorted(fcc_l12_phases.items(), key=lambda x: x[1]['amount'], reverse=True)

            if len(sorted_fcc_l12_phases) < 2:
                warning_message = f"Not enough FCC_L12 phases found."
                logging.warning(f"System {index + 1}: {warning_message}")
                output_results.append({
                    "System Index": index + 1,
                    "System Name": system_name,
                    "Temperature": temperature,
                    "Warning": warning_message
                })
                continue

            phase1, phase2 = sorted_fcc_l12_phases[:2]
            phase1_name, phase1_data = phase1
            phase2_name, phase2_data = phase2

            if (phase1_data['mole_fractions'].get('Al', 0) + phase1_data['mole_fractions'].get('Ti', 0) >
                    phase2_data['mole_fractions'].get('Al', 0) + phase2_data['mole_fractions'].get('Ti', 0)):
                gamma_prime = phase1_name
                gamma = phase2_name
            else:
                gamma_prime = phase2_name
                gamma = phase1_name

            lattice_param_gamma_prime = phase_data[gamma_prime]['lattice_parameter']
            lattice_param_gamma = phase_data[gamma]['lattice_parameter']

            lattice_misfit = 2 * ((lattice_param_gamma_prime - lattice_param_gamma) / (
                    lattice_param_gamma + lattice_param_gamma_prime))

            effective_diffusivities = {}
            for component, values in phase_data[gamma]['diffusivities'].items():
                mole_fraction = phase_data[gamma]['mole_fractions'].get(component, 0)
                effective_diffusivities[component] = mole_fraction * values

            result = {
                "System Index": index + 1,
                "System Name": system_name,
                "Temperature": temperature,
                "Lattice Misfit": lattice_misfit,
                "Amount of Gamma Prime": phase_data[gamma_prime]['amount'],
                "Effective Diffusivity of Gamma Phase": sum(effective_diffusivities.values())
            }

            for element in mass_fractions.keys():
                result[f"Mole Fraction {element}"] = phase_data[gamma]['mole_fractions'].get(element, np.nan)

            output_results.append(result)

        except Exception as e:
            error_message = f"Error processing system {index + 1}: {e}"
            logging.error(error_message)
            output_results.append({
                "System Index": index + 1,
                "System Name": system_name,
                "Temperature": temperature,
                "Error": error_message
            })
            continue

output_df = pd.DataFrame(output_results)

try:
    output_df.to_csv("trail_run_28.csv", index=False)
    logging.info("Results successfully saved to 'trail_run_28.csv'.")
except Exception as e:
    logging.error(f"Error saving output to CSV: {e}")
