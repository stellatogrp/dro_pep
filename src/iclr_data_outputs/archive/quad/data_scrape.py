"""
Data scraping and directory structure creation for Quad experiments.

Similar to lasso/data_scrape.py with support for conditional groups.
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
    if not conditional_groups:
        # No conditional groups, just do regular cartesian product
        return cartesian_product(common_options)
    
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
configs = [
    ['alg=vanilla_gd'],
    # ['pep_obj=obj_val', 'pep_obj=opt_dist_sq_norm'],
    ['pep_obj=obj_val'],
    ['dro_obj=expectation'],
    # ['alpha=0.1'],
    # ['eps=0.01', 'eps=0.1', 'eps=1.0', 'eps=5.0', 'eps=10.0'],
    # ['mu=1'],
    # ['N=20'],
    # ['sgd_iters=500'],
    # ['K_max=[10]', 'K_max=[15]'],
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
    ['eps=1.0'],
    ['training_sample_N=1000'],
]

# Conditional groups: placeholder for future use if configs need conditional dependencies
# Example: {'m=300': ['n=200'], 'm=200': ['n=300']}
Quad_conditional_groups = []

ALL_QUAD_CONFIGS = conditional_product(
    common_options=configs,
    conditional_groups=Quad_conditional_groups
)


# Keys ignored when matching LPEP runs. LPEP does not consume sampled data, so
# a single LPEP run with e.g. training_sample_N=100 is reused across all
# training_sample_N subfolders that share the remaining config keys.
LPEP_PROPAGATE_LIST = ['training_sample_N', 'eps']

# Keys ignored when matching L2O runs. L2O does not use the Wasserstein radius
# `eps`, so a single L2O run is shared across all eps buckets that match on
# the remaining config keys.
L2O_PROPAGATE_LIST = ['eps']


# ==================== Helper Functions ====================

def parse_config_list_to_dict(config_list):
    """
    Convert a config list like ['alg=vanilla_gd', 'mu=0', ...] to a dict
    Handles type coercion for numeric values (int, float)
    """
    config_dict = {}
    for item in config_list:
        key, value = item.split('=', 1)
        # Try to parse as numeric
        try:
            # Check if it's an int
            if '.' not in value:
                config_dict[key] = int(value)
            else:
                config_dict[key] = float(value)
        except ValueError:
            config_dict[key] = value
    return config_dict


def config_matches_directory(config_dict, hydra_config):
    """
    Check if all keys in config_dict match the corresponding values in hydra_config.
    Returns True if all specified config keys match.
    """
    for key, value in config_dict.items():
        if key not in hydra_config:
            return False
        hydra_value = hydra_config[key]
        # Handle type comparison (hydra may load 0 as int, 0.1 as float)
        if isinstance(value, (int, float)) and isinstance(hydra_value, (int, float)):
            if abs(value - hydra_value) > 1e-9:
                return False
        elif value != hydra_value:
            return False
    return True


def get_branching_levels():
    """
    Return the indices of common option levels that have more than one option.
    These create subfolders in the directory structure.
    """
    return [i for i, level in enumerate(configs) if len(level) > 1]


def get_conditional_keys():
    """
    Get all keys that appear in conditional groups.
    These always create subfolders, even if there's only one option for a given base.
    """
    keys = set()
    for group in Quad_conditional_groups:
        for base, dependents in group.items():
            base_key = base.split('=')[0]
            keys.add(base_key)
            for dep in dependents:
                dep_key = dep.split('=')[0]
                keys.add(dep_key)
    return keys


def config_option_to_folder_name(option_str):
    """
    Convert a config option string like 'alg=vanilla_gd' to folder name 'alg_vanilla_gd'
    """
    return option_str.replace('=', '_')


def get_leaf_path(config_list, plots_base='plots'):
    """Get the leaf directory path for a configuration."""
    quad_dir = Path(__file__).parent
    plots_path = quad_dir / plots_base
    branching_levels = get_branching_levels()
    conditional_keys = get_conditional_keys()
    
    # Build path parts
    path_parts = []
    
    # 1. Add branching levels from common options
    for level_idx in branching_levels:
        option_str = config_list[level_idx]
        folder_name = config_option_to_folder_name(option_str)
        path_parts.append(folder_name)
    
    # 2. Add conditional group options (if any)
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


def build_config_to_dirs_map(base_dir='data', propagate_keys=None):
    """
    Build a mapping from each config in ALL_QUAD_CONFIGS to a list of directories
    that match that config.

    Args:
        base_dir: subdirectory under the quad experiment folder to scan.
        propagate_keys: list of config keys to ignore when matching. A run is
            added to EVERY config bucket whose non-propagated keys match the
            run's hydra config. Used for LPEP, which can ignore e.g.
            training_sample_N so one LPEP run is shared across buckets.

    Returns:
        config_to_dirs: dict mapping tuple(config_list) -> list of relative dir paths
        unmatched_dirs: list of directories that don't match any config
    """
    propagate_keys = set(propagate_keys or [])

    quad_dir = Path(__file__).parent
    data_path = quad_dir / base_dir

    if not data_path.exists():
        print(f"Warning: {data_path} does not exist")
        return {tuple(cfg): [] for cfg in ALL_QUAD_CONFIGS}, []

    # Pre-parse all configs to dicts for efficient comparison
    config_dicts = [parse_config_list_to_dict(cfg) for cfg in ALL_QUAD_CONFIGS]
    # Strip propagate_keys for matching; the original config_dicts still own
    # the bucket identity via ALL_QUAD_CONFIGS.
    match_dicts = [
        {k: v for k, v in d.items() if k not in propagate_keys}
        for d in config_dicts
    ]

    # Initialize the mapping: use tuple of config list as key (for hashability)
    config_to_dirs = {tuple(cfg): [] for cfg in ALL_QUAD_CONFIGS}
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
        # When propagate_keys is empty, ALL_QUAD_CONFIGS entries are mutually
        # exclusive on the matched keys, so this still adds each dir once.
        # When propagate_keys is non-empty, a single dir intentionally lands
        # in every bucket whose remaining keys agree.
        for i, match_dict in enumerate(match_dicts):
            if config_matches_directory(match_dict, hydra_config):
                config_to_dirs[tuple(ALL_QUAD_CONFIGS[i])].append(rel_path)
                matched = True

        if not matched:
            unmatched_dirs.append(rel_path)

    return config_to_dirs, unmatched_dirs


def create_plots_directory_structure(config_to_dirs, plots_base='plots', data_dirs_filename='ldro_pep_data_dirs.txt'):
    """
    Create hierarchical folder structure in plots/ based on configs.
    
    Rules:
    - Common options: only create subfolders for options with > 1 choice
    - Conditional group options: ALWAYS create subfolders (both base and dependent)
    - Names folders by replacing '=' with '_'
    - Writes data_dirs.txt with matching directories at each leaf
    """
    quad_dir = Path(__file__).parent
    plots_path = quad_dir / plots_base
    
    # Get branching levels from common options
    branching_levels = get_branching_levels()
    conditional_keys = get_conditional_keys()
    
    print(f"Branching levels (indices): {branching_levels}")
    print(f"Branching level names: {[configs[i][0].split('=')[0] for i in branching_levels]}")
    if conditional_keys:
        print(f"Conditional keys (always create subfolders): {conditional_keys}")
    
    # For each configuration, determine its path and create folder + data_dirs.txt
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
        
        print(f"Created: {leaf_path.relative_to(quad_dir)} -> {data_dirs_filename} ({len(dirs)} dirs)")


if __name__ == "__main__":
    print(f"Number of configurations: {len(ALL_QUAD_CONFIGS)}")
    print()
    
    quad_dir = Path(__file__).parent
    
    # Process LDRO-PEP data (from data/)
    print("=== Processing LDRO-PEP Data (data/) ===")
    ldro_config_to_dirs, ldro_unmatched = build_config_to_dirs_map(base_dir='data')
    create_plots_directory_structure(ldro_config_to_dirs, data_dirs_filename='ldro_pep_data_dirs.txt')
    
    print()
    
    # Process L2O data (from data_l2o/)
    print("=== Processing L2O Data (data_l2o/) ===")
    l2o_data_path = quad_dir / 'data_l2o'
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
    lpep_data_path = quad_dir / 'data_lpep'
    if lpep_data_path.exists():
        lpep_config_to_dirs, lpep_unmatched = build_config_to_dirs_map(
            base_dir='data_lpep',
            propagate_keys=LPEP_PROPAGATE_LIST,
        )
        create_plots_directory_structure(lpep_config_to_dirs, data_dirs_filename='lpep_data_dirs.txt')
    else:
        print(f"  Skipping: {lpep_data_path} does not exist")