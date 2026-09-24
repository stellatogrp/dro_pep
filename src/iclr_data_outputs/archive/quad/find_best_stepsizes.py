"""
Out-of-sample stepsize selection for Quad LDRO-PEP, L2O, and LPEP experiments.

For each configuration:
1. Scan progress.csv files for the row with the minimum `validation_loss`
   (the training code now logs validation loss directly, so no need to re-run
   the validation set here).
2. Across all data-source dirs for a given K, pick the row with the smallest
   `validation_loss` overall.
3. Save to method-specific CSV files:
   - ldro_pep_best_stepsize_schedule.csv
   - l2o_best_stepsize_schedule.csv
   - lpep_best_stepsize_schedule.csv
"""
import sys
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).parent
# Consolidated layout: this file lives at
#   iclr_data_outputs/archive/<problem>/ , so src/ is three levels up,
# where it was two in the original experiment_plots_icml/<problem>/.
src_dir = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from data_scrape import configs, ALL_QUAD_CONFIGS, get_leaf_path


# ==================== Configuration ====================

K_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# ==================== Config Parsing ====================

def parse_config_list_to_dict(config_list):
    """Convert config list like ['alg=vanilla_gd'] to dict."""
    config_dict = {}
    for item in config_list:
        key, value = item.split('=', 1)
        try:
            if '.' not in value:
                config_dict[key] = int(value)
            else:
                config_dict[key] = float(value)
        except ValueError:
            config_dict[key] = value
    return config_dict


def config_option_to_folder_name(option_str):
    """Convert 'alg=vanilla_gd' to 'alg_vanilla_gd'."""
    return option_str.replace('=', '_')


# ==================== Progress CSV Scan ====================

def find_min_validation_row(csv_path, alg, stepsize_type, K):
    """
    Scan a progress.csv and return the row with the smallest `validation_loss`.

    Quad column conventions: vector t columns are `t<k>` for k=0..K-1; vector
    beta columns are `beta<k>` for k=0..K-1 (K values, not K+1).

    Returns:
        (validation_loss, stepsizes) where stepsizes is:
            (t,)              for vanilla_gd
            (t, beta)         for nesterov_fgm
        or None if no valid row could be parsed.
    """
    df = pd.read_csv(csv_path)

    if 'validation_loss' not in df.columns:
        return None

    has_beta = (alg == 'nesterov_fgm')
    is_vector = (stepsize_type == 'vector')

    if is_vector:
        t_cols = [f't{k}' for k in range(K)]
    else:
        t_cols = ['t']

    required_cols = list(t_cols)
    if has_beta:
        beta_cols = [f'beta{k}' for k in range(K)]
        required_cols += beta_cols

    if not all(col in df.columns for col in required_cols):
        return None

    valid = df.dropna(subset=['validation_loss'] + required_cols)
    if valid.empty:
        return None

    best_row = valid.loc[valid['validation_loss'].idxmin()]

    if is_vector:
        t = [float(best_row[c]) for c in t_cols]
    else:
        t = float(best_row['t'])

    if has_beta:
        beta = [float(best_row[c]) for c in beta_cols]
        stepsizes = (t, beta)
    else:
        stepsizes = (t,)

    return float(best_row['validation_loss']), stepsizes


# ==================== Saving ====================

def save_best_stepsize(best_stepsizes, source_dir, K, alg, stepsize_type, output_path):
    """Save best stepsize schedule to CSV, including source directory."""
    has_beta = (alg == 'nesterov_fgm')
    is_vector = (stepsize_type == 'vector')

    data = {'source_dir': [source_dir]}

    t = best_stepsizes[0]
    if is_vector:
        for i in range(K):
            data[f't{i}'] = [float(t[i])]
    else:
        data['t'] = [float(t)]

    if has_beta:
        beta = best_stepsizes[1]
        for i in range(K):
            data[f'beta{i}'] = [float(beta[i])]

    pd.DataFrame(data).to_csv(output_path, index=False)


# ==================== Main Processing ====================

def find_best_for_method(config_dict, leaf_path, quad_dir,
                         data_dirs_file, output_filename, method_name):
    """
    Find best stepsizes for a specific method (LDRO-PEP, L2O, or LPEP) by
    reading `validation_loss` directly from progress.csv files.
    """
    alg = config_dict['alg']
    stepsize_type = 'vector'

    if not data_dirs_file.exists():
        print(f"    [{method_name}] Warning: {data_dirs_file.name} not found, skipping.")
        return

    with open(data_dirs_file, 'r') as f:
        data_dirs = [line.strip() for line in f if line.strip()]

    print(f"    [{method_name}] Found {len(data_dirs)} data source folders")

    for K in K_vals:
        print(f"    [{method_name}] K={K}...")
        best = None  # (validation_loss, stepsizes, source_dir)
        csv_count = 0

        for data_dir in data_dirs:
            progress_csv = quad_dir / data_dir / 'learn_dro_outputs' / f'K_{K}' / 'progress.csv'
            if not progress_csv.exists():
                continue
            csv_count += 1

            result = find_min_validation_row(progress_csv, alg, stepsize_type, K)
            if result is None:
                continue
            val_loss, stepsizes = result

            if best is None or val_loss < best[0]:
                best = (val_loss, stepsizes, data_dir)

        if best is None:
            print(f"      No candidates found (scanned {csv_count} progress.csv files)")
            continue

        best_val_loss, best_stepsizes, best_source_dir = best
        print(f"      Scanned {csv_count} progress.csv files")
        print(f"      Best validation_loss: {best_val_loss:.6f}")
        print(f"      Best source: {best_source_dir}")

        k_output_dir = leaf_path / f'K_{K}'
        k_output_dir.mkdir(parents=True, exist_ok=True)

        output_csv = k_output_dir / output_filename
        save_best_stepsize(best_stepsizes, best_source_dir, K, alg, stepsize_type, output_csv)
        print(f"      Saved to {output_csv.relative_to(quad_dir)}")


def process_configuration(config_list):
    """Process a single configuration: find best stepsizes for each K for LDRO-PEP, L2O, and LPEP."""
    config_dict = parse_config_list_to_dict(config_list)

    alg = config_dict['alg']
    stepsize_type = 'vector'

    quad_dir = Path(__file__).parent

    print(f"\n{'='*60}")
    print(f"Config: {config_list}")
    print(f"  alg={alg}, stepsize_type={stepsize_type}")

    leaf_path = get_leaf_path(config_list)

    print(f"\n  --- LDRO-PEP ---")
    find_best_for_method(
        config_dict, leaf_path, quad_dir,
        leaf_path / 'ldro_pep_data_dirs.txt',
        'ldro_pep_best_stepsize_schedule.csv', 'LDRO-PEP',
    )

    print(f"\n  --- L2O ---")
    find_best_for_method(
        config_dict, leaf_path, quad_dir,
        leaf_path / 'l2o_data_dirs.txt',
        'l2o_best_stepsize_schedule.csv', 'L2O',
    )

    print(f"\n  --- LPEP ---")
    find_best_for_method(
        config_dict, leaf_path, quad_dir,
        leaf_path / 'lpep_data_dirs.txt',
        'lpep_best_stepsize_schedule.csv', 'LPEP',
    )


def main():
    print("="*60)
    print("Out-of-Sample Stepsize Selection (from progress.csv validation_loss) - Quad")
    print("="*60)
    print(f"Total configurations: {len(ALL_QUAD_CONFIGS)}")
    print(f"K values: {K_vals}")

    for i, config in enumerate(ALL_QUAD_CONFIGS):
        print(f"\n[{i+1}/{len(ALL_QUAD_CONFIGS)}] Processing config...")
        process_configuration(config)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
