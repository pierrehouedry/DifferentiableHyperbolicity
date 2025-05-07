import argparse
import yaml
from tabulate import tabulate
from hyperbolicity.tree_fitting_methods.hyperbolicity_learning import train_distance_matrix
from dataclasses import dataclass, field, fields
from hyperbolicity.utils import construct_weighted_matrix
from hyperbolicity.tree_fitting_methods.gromov import gromov_tree
from typing import List, Any
import pickle
import torch
import time
import datetime
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

@dataclass
class GridSearchConfig:
    learning_rates: List[float]
    distance_regs: List[float]
    scale_delta: List[float]
    epochs: List[int]
    batch_size: List[List[int]]
    dataset: str

    @classmethod
    def from_dict(cls, data: dict):
        """Creates an instance from a dictionary, validating keys and types."""
        expected_keys = {f.name for f in fields(cls)}
        received_keys = set(data.keys())

        missing_keys = expected_keys - received_keys
        if missing_keys:
            raise ValueError(f"Missing configuration keys: {', '.join(missing_keys)}")

        extra_keys = received_keys - expected_keys
        if extra_keys:
            raise ValueError(f"Unexpected configuration keys: {', '.join(extra_keys)}")

        # Basic type validation happens during dataclass instantiation
        try:
            # Type hints help, but complex types like List[List[int]] might need more checks if needed
            return cls(**data)
        except TypeError as e:
            # Catch potential type errors during instantiation
            raise ValueError(f"Configuration type error: {e}")


def main(config: GridSearchConfig):
    print("\nStarting grid search with validated config:")
    print(f"Learning Rates: {config.learning_rates}")
    print(f"Distance Regs: {config.distance_regs}")

    # Load dataset
    try:
        with open(config.dataset, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.dataset}")
        exit(1)
    except pickle.UnpicklingError as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the dataset: {e}")
        exit(1)
    print(f"Dataset loaded successfully from {config.dataset}")

    # Convert dataset to torch float32
    distances = torch.tensor(dataset, dtype=torch.float64)

    #####
    new_row = torch.full((1, distances.shape[1]), 20)
    distances = torch.cat((distances, new_row), dim=0)
    new_column = torch.full((distances.shape[0], 1), 20)
    distances = torch.cat((distances, new_column), dim=1)
    distances[-1,-1] = 0
    #####

    # Generate the folder name once based on a timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"results_expes/{config.dataset.split('/')[-1].split('.')[0]}_{timestamp}/"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder {folder_name} already exists. Reusing it.")

    # Create results.csv if it doesn't exist
    results_file = os.path.join(folder_name, "results.csv")
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("learning_rate, distance_reg, scale_delta,epochs, batch_size,n_batches, intermediate_distortion, intermediate_l1, mean_optim_l1, min_optim_l1, std_optim_l1, mean_optim_distortion, min_optim_distortion, std_optim_distortion, epochs_reached\n")
    else:
        print(f"Results file already exists at {results_file}. Appending results.")

    # Create all combinations of hyperparameters
    hyperparameter_combinations = [
        (float(lr), float(dr), float(sd), epoch, batch)
        for lr in config.learning_rates
        for dr in config.distance_regs
        for sd in config.scale_delta
        for epoch in config.epochs
        for batch in config.batch_size
    ]
    print(f"Total combinations: {len(hyperparameter_combinations)}")
    pbar = tqdm(total=len(hyperparameter_combinations), desc="Grid Search Progress", unit="combination")
    pbar.set_postfix_str("Starting...")
    for i, (lr, dr, sd, epoch, batch) in enumerate(hyperparameter_combinations):
        pbar.set_postfix_str(f"LR={lr}, DR={dr}, SD={sd}, Epochs={epoch}, Batch Size={batch}")
    
        # Call the training function with the current combination
        best_weights, losses, deltas, errors, duration = train_distance_matrix(
            distances,
            learning_rate=lr,
            distance_reg=dr,
            scale_delta=sd,
            num_epochs=epoch,
            batch_size=batch[0],
            n_batches=batch[1],
            gpu=True,
            verbose=True,
        )

        # Save state dict
        if not torch.isnan(best_weights).any():
            state_dict = {
            "weights": best_weights,
            "losses": losses,
            "deltas": deltas,
            "errors": errors,
            "duration": duration,
            }

            # Save state dict
            file_name = f"{folder_name}lr_{lr}_dr_{dr}_sd_{sd}_epoch_{epoch}_batch_{batch[0]}_n_batches_{batch[1]}.pt"
            torch.save(state_dict, file_name)

            # Compute scores
            num_nodes = distances.shape[0]
            np.random.seed(42)
            indices = np.random.choice(num_nodes, 100, replace=False)
            indices = [num_nodes-1]*5
            #denom = num_nodes * (num_nodes - 1)
            edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
            distance_optimized = construct_weighted_matrix(best_weights, num_nodes, edges)
            intermediate_distortion = torch.abs(distance_optimized - distances).max().item()
            intermediate_l1 = torch.abs(distance_optimized - distances).mean().item()
            optim_l1 = []
            optim_distortion = []
            distance_optimized_cpu = distance_optimized.cpu().numpy()
            distances_cpu = distances.cpu().numpy()
            for j in indices:
                T_opt = gromov_tree(distance_optimized_cpu, j)
                optim_distortion.append(np.abs(T_opt - distances_cpu).max())
                optim_l1.append(np.abs(T_opt - distances_cpu).mean())

            # Append results to csv
            with open(results_file, 'a') as f:
                f.write(f"{lr},{dr},{sd},{epoch},{batch[0]},{batch[1]},{intermediate_distortion},{intermediate_l1},{np.mean(optim_l1)},{np.min(optim_l1)},{np.std(optim_l1)},{np.mean(optim_distortion)},{np.min(optim_distortion)},{np.std(optim_distortion)},{len(losses)}\n")
        else:
            with open(results_file, 'a') as f:
                f.write(f"{lr},{dr},{sd},{epoch},{batch[0]},{batch[1]},nan,nan,nan,nan,nan,nan,nan,nan,{len(losses)}\n")
        pbar.update(1)
        pbar.close()

    print(f"Grid search completed. Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search based on a config file.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="../configs/grid_search.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="../datasets/D_celegan.pkl",
    )

    args = parser.parse_args()

    config_dict = None
    try:
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

    if not config_dict:
        print("Config file is empty.")
        exit(1)

    try:
        # Validate and create the config object
        config = GridSearchConfig.from_dict(config_dict)

        print("Configuration Parameters (Validated):")
        # Prepare data for tabulate using the validated config object
        table_data = [[f.name, getattr(config, f.name)] for f in fields(config)]
        print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

        # Pass the validated config object to main
        main(config)

    except ValueError as e:
        print(f"Error validating configuration: {e}")
        exit(1)
    except Exception as e: # Catch other potential errors during validation/instantiation
        print(f"An unexpected error occurred during configuration processing: {e}")
        exit(1)