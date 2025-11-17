import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14,
    "figure.figsize": (10, 6),
})

pep_K_max = 30
exp_K_max = 30
cvar_K_max = 30

GD_pep = pd.read_csv('data/pep/grad_desc_1_30/pep.csv')
NGD_pep = pd.read_csv('data/pep/nesterov_fgm_1_30/pep.csv')

GD_exp_dro = pd.read_csv('data/dro/grad_desc_exp_1_30/dro.csv')
GD_cvar_dro = pd.read_csv('data/dro/grad_desc_cvar_1_30/dro.csv')
NGD_exp_dro = pd.read_csv('data/dro/nesterov_fgm_exp_1_30/dro.csv')
NGD_cvar_dro = pd.read_csv('data/dro/nesterov_fgm_cvar_1_30/dro.csv')

PEP_OBJ = 'opt_dist_sq_norm'


def plot_times():
    GD_color = 'tab:blue'
    NGD_color = 'tab:orange'

    GD_pep_times = GD_pep[GD_pep['obj'] == PEP_OBJ]['solvetime'][:pep_K_max]
    NGD_pep_times = NGD_pep[NGD_pep['obj'] == PEP_OBJ]['solvetime'][:pep_K_max]

    GD_exp_times = GD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    GD_cvar_times = GD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_exp_times = NGD_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    NGD_cvar_times = NGD_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]

    # ax[0].legend()

    fig, ax = plt.subplots(1, 3)

    ax[0].set_ylabel('Solve time(s)')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # ax[1].set_yscale('log')
    # ax[0].set_xscale('log')
    # ax[1].set_xscale('log')

    ax[0].grid(color='lightgray', alpha=0.3)
    ax[1].grid(color='lightgray', alpha=0.3)
    ax[2].grid(color='lightgray', alpha=0.3)

    ax[0].set_xticks([10, 20, 30])

    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])

    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    ax[0].set_title('Worst-case')
    ax[2].set_title('Expectation')
    ax[1].set_title('CVaR')

    ax[0].plot(range(1, exp_K_max + 1), GD_pep_times, label='GD', color=GD_color)
    ax[0].plot(range(1, exp_K_max + 1), NGD_pep_times, label='FGM', color=NGD_color)
    
    ax[2].plot(range(1, exp_K_max + 1), GD_exp_times, label='GD', color=GD_color)
    ax[2].plot(range(1, exp_K_max + 1), NGD_exp_times, label='FGM', color=NGD_color)

    ax[1].plot(range(1, cvar_K_max + 1), GD_cvar_times, label='GD', color=GD_color)
    ax[1].plot(range(1, cvar_K_max + 1), NGD_cvar_times, label='FGM', color=NGD_color)

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.05, box.width, box.height-.05])

    # ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4)

    plt.suptitle('Strongly Convex Quadratic Minimization, Solve times')

    # plt.show()

    # plt.savefig('quad_strongcvx_times.pdf')

    # 1. Create a common K column (1 to 30)
    k_values = np.arange(1, pep_K_max + 1)

    # 2. Create DataFrame for GD
    # We use .values to strip the original indices and align the data
    df_gd = pd.DataFrame({
        'K': k_values,
        'worst-case (PEP)': GD_pep_times.values,
        'cvar': GD_cvar_times.values,
        'expectation': GD_exp_times.values,
        'algorithm': 'GD'
    })

    # 3. Create DataFrame for FGM (NGD)
    df_fgm = pd.DataFrame({
        'K': k_values,
        'worst-case (PEP)': NGD_pep_times.values,
        'cvar': NGD_cvar_times.values,
        'expectation': NGD_exp_times.values,
        'algorithm': 'FGM'
    })

    # 4. Combine the DataFrames
    df_combined = pd.concat([df_gd, df_fgm], ignore_index=True)

    # 5. Filter for K in increments of 5
    df_filtered = df_combined[df_combined['K'] % 5 == 0].copy()

    # 6. Reorder columns to the desired format
    final_columns = ['algorithm', 'K', 'worst-case (PEP)', 'cvar', 'expectation']
    df_final = df_filtered[final_columns].copy() # Use .copy() to be explicit

    # 7. Set algorithm to blank '' for all but the first row in each group
    df_final.loc[df_final['algorithm'].duplicated(), 'algorithm'] = ''

    # --- START: New formatting logic ---
    
    def custom_format_to_string(x):
        """
        Rounds to 2 decimal places, unless the result is 0.00,
        in which case it rounds to the first significant digit.
        Returns a string representation.
        """
        # Handle NaN, None, or other non-numeric types gracefully
        if not isinstance(x, (int, float)) or pd.isna(x):
            return "" 
        
        if x == 0:
            return "0.00"
        
        # Round to 2 decimal places
        x_rounded_2dp = round(x, 2)
        
        if x_rounded_2dp != 0.0:
            # Standard case (e.g., 1.234 -> "1.23" or 0.007 -> "0.01")
            return f"{x_rounded_2dp:.2f}"
        else:
            # Special case: x is very small (e.g., 0.0001)
            n = -int(np.floor(np.log10(np.abs(x))))
            x_rounded_1sigfig = round(x, n)
            return f"{x_rounded_1sigfig:.{n}f}"
    
    # Apply the custom string formatting to the time columns
    # Use .applymap() to apply the function to all elements in the selected columns
    time_cols = ['worst-case (PEP)', 'cvar', 'expectation']
    df_final[time_cols] = df_final[time_cols].map(custom_format_to_string)
    
    # --- END: New formatting logic ---
    # 8. Save the final DataFrame to a CSV file
    df_final.to_csv('times_quad_strongcvx.csv', index=False, float_format='%.6f')


if __name__ == '__main__':
    plot_times()
