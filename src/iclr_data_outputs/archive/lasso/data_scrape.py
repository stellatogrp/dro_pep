"""
Data scraping and directory structure creation for Lasso experiments.

Similar to quad/data_scrape.py but with support for conditional groups.
Conditional group options always create nested subfolders, even if there's only one option.
"""
from itertools import product
from pathlib import Path
from omegaconf import OmegaConf


def cartesian_product(options):
    return [list(combo) for combo in product(*options)]


def conditional_product(common_options, conditional_groups):
    """
    Create cartesian product with conditional dependencies.
    
    Args:
        common_options: List of lists - options included in ALL combinations
        conditional_groups: List of dicts mapping base option to dependent options
    
    Returns:
        List of lists representing all valid combinations
    """
    # First, expand each conditional group into (base, dependent) pairs
    conditional_pairs = []
    for group in conditional_groups:
        pairs = []
        for base, dependents in group.items():
            for dep in dependents:
                pairs.append([base, dep])  # Each pair becomes one "option" in the product
        conditional_pairs.append(pairs)
    
    # Now cartesian product: common_options × flattened conditional pairs
    all_options = common_options + conditional_pairs
    
    results = []
    for combo in product(*all_options):
        # Flatten: some elements are strings (from common), some are lists (from conditional)
        flat = []
        for item in combo:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        results.append(flat)
    
    return results


# ==================== Configuration ====================

# Common options (only create subfolders for options with > 1 choice)
Lasso_options = [
    ['alg=ista'],
    # ['training_sample_N=100',
    #  'training_sample_N=200',
    #  'training_sample_N=300',
    #  'training_sample_N=400',
    #  'training_sample_N=500',
    #  'training_sample_N=600',
    #  'training_sample_N=700',
    #  'training_sample_N=800',
    #  'training_sample_N=900',
    #  'training_sample_N=1000'],
    ['training_sample_N=1000'],
    ['eps=10.0'],
    # ['dro_obj=expectation'],
    # ['alpha=0.1'],
]

# Conditional groups: m and n are tied together
# Both m and n should always create subfolders since they're conditional
Lasso_conditional_groups = [
    # {
    #     'm=300': ['n=200'],
    #     'm=200': ['n=300'],
    # },
]

ALL_LASSO_CONFIGS = conditional_product(
    common_options=Lasso_options,
    conditional_groups=Lasso_conditional_groups
)

# We also export just the configs list for consistency with quad
configs = Lasso_options


# Keys ignored when matching LPEP runs. LPEP does not consume sampled data, so
# a single LPEP run with e.g. training_sample_N=100 is reused across all
# training_sample_N subfolders that share the remaining config keys.
LPEP_PROPAGATE_LIST = ['training_sample_N', 'eps']

# Keys ignored when matching L2O runs. L2O does not use the Wasserstein radius
# `eps`, so a single L2O run is shared across all eps buckets that match on
# the remaining config keys.
L2O_PROPAGATE_LIST = ['eps']

# L2O-ALISTA shares the same propagate semantics as L2O.
L2O_ALISTA_PROPAGATE_LIST = L2O_PROPAGATE_LIST


# ==================== Helper Functions ====================

def parse_config_list_to_dict(config_list):
    """Convert config list like ['alg=ista', 'm=300', 'n=200'] to dict."""
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


def config_matches_directory(config_dict, hydra_config):
    """Check if a config_dict matches a Hydra config loaded from a directory."""
    for key, expected_value in config_dict.items():
        if key not in hydra_config:
            return False
        actual_value = hydra_config[key]
        if isinstance(expected_value, float):
            if abs(actual_value - expected_value) > 1e-9:
                return False
        elif actual_value != expected_value:
            return False
    return True


def get_branching_levels():
    """
    Return the indices of common option levels that have more than one option.
    These create subfolders in the directory structure.
    """
    return [i for i, level in enumerate(Lasso_options) if len(level) > 1]


def get_conditional_keys():
    """
    Get all keys that appear in conditional groups.
    These always create subfolders, even if there's only one option for a given base.
    """
    keys = set()
    for group in Lasso_conditional_groups:
        for base, dependents in group.items():
            base_key = base.split('=')[0]
            keys.add(base_key)
            for dep in dependents:
                dep_key = dep.split('=')[0]
                keys.add(dep_key)
    return keys


def config_option_to_folder_name(option_str):
    """Convert a config option string like 'alg=ista' to folder name 'alg_ista'."""
    return option_str.replace('=', '_')


def get_config_level_index(config_list, key):
    """Find the index of a key in the config list."""
    for i, item in enumerate(config_list):
        if item.startswith(f'{key}='):
            return i
    return -1


def get_leaf_path(config_list, plots_base='plots'):
    """Get the leaf directory path for a configuration."""
    lasso_dir = Path(__file__).parent
    plots_path = lasso_dir / plots_base
    branching_levels = get_branching_levels()
    conditional_keys = get_conditional_keys()
    
    # Build path parts
    path_parts = []
    
    # 1. Add branching levels from common options
    for level_idx in branching_levels:
        option_str = config_list[level_idx]
        folder_name = config_option_to_folder_name(option_str)
        path_parts.append(folder_name)
    
    # 2. Add conditional group options
    for key in sorted(conditional_keys):
        for item in config_list:
            if item.startswith(f'{key}='):
                folder_name = config_option_to_folder_name(item)
                path_parts.append(folder_name)
                break
    
    # Create the full path
    leaf_path = plots_path
    for part in path_parts:
        leaf_path = leaf_path / part
    return leaf_path


def create_plots_directory_structure(config_to_dirs, plots_base='plots', data_dirs_filename='ldro_pep_data_dirs.txt'):
    """
    Create hierarchical folder structure in plots/ based on configs.
    
    Rules:
    - Common options: only create subfolders for options with > 1 choice
    - Conditional group options: ALWAYS create subfolders (both base and dependent)
    - Names folders by replacing '=' with '_'
    - Writes data_dirs.txt with matching directories at each leaf
    """
    lasso_dir = Path(__file__).parent
    plots_path = lasso_dir / plots_base
    
    # Get branching levels from common options
    branching_levels = get_branching_levels()
    conditional_keys = get_conditional_keys()
    
    print(f"Common branching levels (indices): {branching_levels}")
    print(f"Common branching level names: {[Lasso_options[i][0].split('=')[0] for i in branching_levels]}")
    print(f"Conditional keys (always create subfolders): {conditional_keys}")
    
    # For each configuration, determine its path and create folder
    for config_tuple, dirs in config_to_dirs.items():
        config_list = list(config_tuple)
        
        # Get the leaf path for this config
        leaf_path = get_leaf_path(config_list, plots_base)
        
        # Create directory (including parents)
        leaf_path.mkdir(parents=True, exist_ok=True)
        
        # Write data_dirs.txt with the matching directories
        data_dirs_file = leaf_path / data_dirs_filename
        with open(data_dirs_file, 'w') as f:
            for d in sorted(dirs):
                f.write(f"{d}\n")
        
        print(f"Created: {leaf_path.relative_to(lasso_dir)} -> {data_dirs_filename} ({len(dirs)} dirs)")
    
    print(f"\nTotal configurations: {len(ALL_LASSO_CONFIGS)}")


def build_config_to_dirs_map(base_dir='data', propagate_keys=None):
    """
    Build a mapping from each config in ALL_LASSO_CONFIGS to a list of directories
    that match that config.

    Args:
        base_dir: subdirectory under the lasso experiment folder to scan.
        propagate_keys: list of config keys to ignore when matching. A run is
            added to EVERY config bucket whose non-propagated keys match the
            run's hydra config. Used for LPEP, which ignores e.g.
            training_sample_N, so one LPEP run is shared across all
            training_sample_N subfolders.

    Returns:
        config_to_dirs: dict mapping tuple(config_list) -> list of relative dir paths
        unmatched_dirs: list of directories that don't match any config
    """
    propagate_keys = set(propagate_keys or [])

    lasso_dir = Path(__file__).parent
    data_path = lasso_dir / base_dir

    if not data_path.exists():
        print(f"Warning: {data_path} does not exist")
        return {tuple(cfg): [] for cfg in ALL_LASSO_CONFIGS}, []

    # Pre-parse all configs to dicts for efficient comparison
    config_dicts = [parse_config_list_to_dict(cfg) for cfg in ALL_LASSO_CONFIGS]
    # Strip propagate_keys for matching; the original config_dicts still own
    # the bucket identity via ALL_LASSO_CONFIGS.
    match_dicts = [
        {k: v for k, v in d.items() if k not in propagate_keys}
        for d in config_dicts
    ]

    # Initialize the mapping
    config_to_dirs = {tuple(cfg): [] for cfg in ALL_LASSO_CONFIGS}
    unmatched_dirs = []

    # Scan all timestamped directories
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue

        hydra_config_path = subdir / '.hydra' / 'config.yaml'
        if not hydra_config_path.exists():
            continue

        try:
            hydra_config = OmegaConf.load(hydra_config_path)
        except Exception as e:
            print(f"Warning: Could not load {hydra_config_path}: {e}")
            continue

        rel_path = f"{base_dir}/{subdir.name}"
        matched = False
        # When propagate_keys is empty, ALL_LASSO_CONFIGS entries are mutually
        # exclusive on the matched keys, so this still adds each dir once.
        # When propagate_keys is non-empty, a single dir intentionally lands
        # in every bucket whose remaining keys agree.
        for i, match_dict in enumerate(match_dicts):
            if config_matches_directory(match_dict, hydra_config):
                config_to_dirs[tuple(ALL_LASSO_CONFIGS[i])].append(rel_path)
                matched = True

        if not matched:
            unmatched_dirs.append(rel_path)

    return config_to_dirs, unmatched_dirs


if __name__ == "__main__":
    print(f"Number of configurations: {len(ALL_LASSO_CONFIGS)}")
    print("\nSample configurations:")
    for i, cfg in enumerate(ALL_LASSO_CONFIGS[:4]):
        print(f"  {i}: {cfg}")
    print()

    lasso_dir = Path(__file__).parent

    # Process LDRO-PEP data (from data/)
    print("=== Processing LDRO-PEP Data (data/) ===")
    ldro_config_to_dirs, ldro_unmatched = build_config_to_dirs_map(base_dir='data')
    create_plots_directory_structure(ldro_config_to_dirs, data_dirs_filename='ldro_pep_data_dirs.txt')

    print()

    # Process L2O data (from data_l2o/)
    print("=== Processing L2O Data (data_l2o/) ===")
    l2o_data_path = lasso_dir / 'data_l2o'
    if l2o_data_path.exists():
        l2o_config_to_dirs, l2o_unmatched = build_config_to_dirs_map(
            base_dir='data_l2o',
            propagate_keys=L2O_PROPAGATE_LIST,
        )
        create_plots_directory_structure(l2o_config_to_dirs, data_dirs_filename='l2o_data_dirs.txt')
    else:
        print(f"  Skipping: {l2o_data_path} does not exist")

    print()

    # Process LPEP data (from data_lpep/)
    print("=== Processing LPEP Data (data_lpep/) ===")
    lpep_data_path = lasso_dir / 'data_lpep'
    if lpep_data_path.exists():
        lpep_config_to_dirs, lpep_unmatched = build_config_to_dirs_map(
            base_dir='data_lpep',
            propagate_keys=LPEP_PROPAGATE_LIST,
        )
        create_plots_directory_structure(lpep_config_to_dirs, data_dirs_filename='lpep_data_dirs.txt')
    else:
        print(f"  Skipping: {lpep_data_path} does not exist")

    print()

    # Process L2O-ALISTA data (from data_alista/) — same propagate semantics as L2O.
    print("=== Processing L2O-ALISTA Data (data_alista/) ===")
    alista_data_path = lasso_dir / 'data_alista'
    if alista_data_path.exists():
        alista_config_to_dirs, alista_unmatched = build_config_to_dirs_map(
            base_dir='data_alista',
            propagate_keys=L2O_ALISTA_PROPAGATE_LIST,
        )
        create_plots_directory_structure(alista_config_to_dirs, data_dirs_filename='l2o_alista_data_dirs.txt')
    else:
        print(f"  Skipping: {alista_data_path} does not exist")