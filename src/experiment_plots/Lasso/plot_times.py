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

pep_K_max = 25
exp_K_max = 25
cvar_K_max = 25

ISTA_samples = pd.read_csv('data/samples/ISTA_1_25/samples.csv')
FISTA_samples = pd.read_csv('data/samples/FISTA_1_25/samples.csv')
# OptISTA_samples = pd.read_csv('data/samples/OptISTA_1_25/samples.csv')

ISTA_pep = pd.read_csv('data/pep/ISTA_1_25/pep.csv')
FISTA_pep = pd.read_csv('data/pep/FISTA_1_25/pep.csv')
# OptISTA_pep = pd.read_csv('data/pep/OptISTA_1_25/pep.csv')

ISTA_exp_dro = pd.read_csv('data/dro/ISTA_exp_1_25/dro.csv')
ISTA_cvar_dro = pd.read_csv('data/dro/ISTA_cvar_1_25/dro.csv')
FISTA_exp_dro = pd.read_csv('data/dro/FISTA_exp_1_25/dro.csv')
FISTA_cvar_dro = pd.read_csv('data/dro/FISTA_cvar_1_25/dro.csv')


def plot_times():
    ISTA_color = 'tab:blue'
    FISTA_color = 'tab:green'
    # OptISTA_color = 'tab:red'

    ISTA_pep_times = ISTA_pep[ISTA_pep['obj'] == 'obj_val']['solvetime'][:pep_K_max]
    FISTA_pep_times = FISTA_pep[FISTA_pep['obj'] == 'obj_val']['solvetime'][:pep_K_max]
    # OptISTA_pep_times = OptISTA_pep[OptISTA_pep['obj'] == 'obj_val']['solvetime'][:pep_K_max]

    ISTA_exp_times = ISTA_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    ISTA_cvar_times = ISTA_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    FISTA_exp_times = FISTA_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    FISTA_cvar_times = FISTA_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    # OptISTA_exp_times = OptISTA_exp_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]
    # OptISTA_cvar_times = OptISTA_cvar_dro.groupby(['K'])['solvetime'].mean().iloc[:exp_K_max]

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

    ax[0].set_xticks([5, 10, 15, 20, 25])

    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])

    ax[0].set_xlabel(r'$K$')
    ax[1].set_xlabel(r'$K$')

    ax[0].set_title('Worst-case')
    ax[2].set_title('Expectation')
    ax[1].set_title('CVar')

    ax[0].plot(range(1, exp_K_max + 1), ISTA_pep_times, label='ISTA', color=ISTA_color)
    ax[0].plot(range(1, exp_K_max + 1), FISTA_pep_times, label='FISTA', color=FISTA_color)
    # ax[0].plot(range(1, exp_K_max + 1), OptISTA_pep_times, label='OptISTA', color=OptISTA_color)

    ax[2].plot(range(1, exp_K_max + 1), ISTA_exp_times, label='ISTA', color=ISTA_color)
    ax[2].plot(range(1, exp_K_max + 1), FISTA_exp_times, label='FISTA', color=FISTA_color)
    # ax[2].plot(range(1, exp_K_max + 1), OptISTA_exp_times, label='OptISTA', color=OptISTA_color)

    ax[1].plot(range(1, exp_K_max + 1), ISTA_cvar_times, label='ISTA', color=ISTA_color)
    ax[1].plot(range(1, exp_K_max + 1), FISTA_cvar_times, label='FISTA', color=FISTA_color)
    # ax[1].plot(range(1, exp_K_max + 1), OptISTA_cvar_times, label='OptISTA', color=OptISTA_color)

    for axi in ax:
        box = axi.get_position()
        # x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        axi.set_position([box.x0, box.y0+.05, box.width, box.height-.05])

    # ax[0].legend()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4)

    plt.suptitle('Lasso, Solve times')

    # plt.show()

    # plt.savefig('Lasso_times.pdf')

    k_values = np.arange(1, pep_K_max + 1)

    # 2. Create DataFrame for ISTA
    # We use .values to strip the original indices and align the data
    df_ISTA = pd.DataFrame({
        'K': k_values,
        'worst-case (PEP)': ISTA_pep_times.values,
        'cvar': ISTA_cvar_times.values,
        'expectation': ISTA_exp_times.values,
        'algorithm': 'ISTA'
    })

    # 3. Create DataFrame for FISTA (FISTA)
    df_FISTA = pd.DataFrame({
        'K': k_values,
        'worst-case (PEP)': FISTA_pep_times.values,
        'cvar': FISTA_cvar_times.values,
        'expectation': FISTA_exp_times.values,
        'algorithm': 'FISTA'
    })

    # 4. Combine the DataFrames
    df_combined = pd.concat([df_ISTA, df_FISTA], ignore_index=True)

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
    df_final.to_csv('times_lasso.csv', index=False, float_format='%.6f')


if __name__ == '__main__':
    plot_times()
