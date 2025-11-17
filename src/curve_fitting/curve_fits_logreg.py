import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 14, # default: 14
    "figure.figsize": (9, 6), # default: (9, 6)
    "lines.linewidth": 2.5,
})


def curve_fit(
        data,
        csv_fname=None,
        plot_fname=None,
        mincutoff=1,
        c=1.0,
        rho_fix=True, rho_val=1.0,
        gamma_fix=True, gamma_val=0.0,
        eta_fix=True, eta_val=0.0,
        log_scale=True,
    ):
    if len(data.shape) == 1 :
        K = np.arange(1,len(data)+1)
        phi = data
    else :
        K = data['K'].to_numpy()
        phi = data['dro_feas_sol'].to_numpy()

    K_vals = K.reshape(-1, 1)
    A = np.hstack([np.ones(K_vals.shape), K_vals, -np.log(K_vals+c), np.log(np.log(K_vals+c))])
    b = np.log(phi/phi[0]/np.e)
    A = A[mincutoff-1:, :]
    b = b[mincutoff-1:]

    b_inv = np.abs(1/b)

    # log (fit) = log(C) + K log(rho) + (gamma) (-log(K+c)) + (eta) log(log(K+c))
    x = cp.Variable(A.shape[1]) # x = [ log(C), log(rho), (gamma), (eta) ]
    
    # minimize gap with interpolation
    obj = .5 * cp.sum_squares(cp.multiply(b_inv, A @ x - b))
    constraints = []

    constraints += [A @ x >= b]
    constraints += [x[1] <= np.log(1.0)]
    constraints += [x[2] >= 0.0]
    constraints += [x[3] >= 0.0]

    if rho_fix:
        constraints += [x[1] == np.log(rho_val)]
    
    if gamma_fix:
        constraints += [x[2] == gamma_val]
    
    if eta_fix:
        constraints += [x[3] == eta_val]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

    C = np.exp(x.value[0]) * phi[0] * np.e
    rho = np.exp(x.value[1])
    gamma = x.value[2]
    eta = x.value[3]

    fitted_vals = C * np.power(rho, K) * np.power(K+c, -gamma) * np.power(np.log(K+1), eta)

    fig, ax = plt.subplots()
    if log_scale :
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(K, phi, label='DRO objective', color='#1E88E5')
    ax.plot(K, fitted_vals, label='fitted curve', linestyle='--', color='tab:gray', linewidth=1.5)
    if np.abs(eta) <= 1e-6 :
        ax.set_title(rf'(contraction) $\rho = {rho:.4f}$ (sublinear) $\gamma = {gamma:.4f}$')
    else :
        ax.set_title(rf'(contraction) $\rho = {rho:.4f}$ (sublinear) $\gamma = {gamma:.4f}$ (log) $\eta = {eta:.4f}$')
    ax.grid(color='lightgray', alpha=0.3)
    ax.legend()

    if csv_fname is not None:
        data = [
            {'C': C, 'rho': rho, 'gamma': gamma, 'eta': eta},
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_fname, index=False)
    if plot_fname is not None:
        plt.savefig(plot_fname)


def plot_fits(
        dro_data,
        csv_files,
        plot_fname=None, plot_title=None,
        sample_data=None,
        log_scale=True,
    ) :
    assert len(dro_data) == len(csv_files)

    fig, ax = plt.subplots()
    if log_scale:
        ax.set_xscale('log')
    ax.set_yscale('log')
    if plot_title is not None:
        ax.set_title(plot_title)

    for (dd, cf) in zip(dro_data, csv_files) :
        if len(dd.shape) == 1 :
            K = np.arange(1,len(dd)+1)
            phi = dd
        else :
            K = dd['K'].to_numpy()
            phi = dd['dro_feas_sol'].to_numpy()

        df = pd.read_csv(cf)
        rho = df['rho'].values[0]
        gamma = df['gamma'].values[0]
        C = df['C'].values[0]
        eta = df['eta'].values[0]
        fitted_vals = C * np.power(rho, K) * np.power(K+1, -gamma) * np.power(np.log(K+1), eta)
        
        if np.abs(eta) <= 1e-6 :
            ax.plot(K, phi, label=rf'$(\rho, \gamma)$ = ({rho:.6f}, {gamma:.6f})')
        else :
            ax.plot(K, phi, label=rf'$(\rho, \gamma, \eta)$ = ({rho:.6f}, {gamma:.6f}, {eta:.6f})')
        ax.plot(K, fitted_vals, linestyle='--', color='tab:gray', linewidth=1.5)

    if sample_data is not None :
        for sd in sample_data:
            ax.plot(K, sd, color='k', linewidth=1.0)
    
    ax.grid(color='lightgray', alpha=0.3)
    ax.legend()

    if plot_fname is not None:
        plt.savefig(plot_fname)
    else:
        plt.show()


def main_logreg_exp():
    LogReg_gd_exp = pd.read_csv(f'LogReg/dro/grad_desc_exp_1_30/dro.csv')
    LogReg_gd_exp_data = LogReg_gd_exp[LogReg_gd_exp['eps_idx'] == 2]
    LogReg_fgm_exp = pd.read_csv(f'LogReg/dro/nesterov_fgm_exp_1_30/dro.csv')
    LogReg_fgm_exp_data = LogReg_fgm_exp[LogReg_fgm_exp['eps_idx'] == 0]

    curve_fit(
        LogReg_gd_exp_data,
        csv_fname='LogReg/fits/gd_exp_fit_params.csv',
        plot_fname='LogReg/fits/gd_exp_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        log_scale=False,
    )

    curve_fit(
        LogReg_fgm_exp_data,
        csv_fname='LogReg/fits/fgm_exp_fit_params.csv',
        plot_fname='LogReg/fits/fgm_exp_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        log_scale=False,
    )

    plot_fits(
        dro_data = [
            LogReg_gd_exp_data,
            LogReg_fgm_exp_data,
        ],
        csv_files = [
            'LogReg/fits/gd_exp_fit_params.csv',
            'LogReg/fits/fgm_exp_fit_params.csv',
        ],
        plot_fname='LogReg/exp_fit.pdf',
        plot_title=r'Logistic Regression: Expectation',
        log_scale=False,
    )

def main_logreg_cvar():
    LogReg_gd_cvar = pd.read_csv(f'LogReg/dro/grad_desc_cvar_1_30/dro.csv')
    LogReg_gd_cvar_data = LogReg_gd_cvar[LogReg_gd_cvar['eps_idx'] == 0]
    LogReg_fgm_cvar = pd.read_csv(f'LogReg/dro/nesterov_fgm_cvar_1_30/dro.csv')
    LogReg_fgm_cvar_data = LogReg_fgm_cvar[LogReg_fgm_cvar['eps_idx'] == 0]

    curve_fit(
        LogReg_gd_cvar_data,
        csv_fname='LogReg/fits/gd_cvar_fit_params.csv',
        plot_fname='LogReg/fits/gd_cvar_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        log_scale=False,
    )

    curve_fit(
        LogReg_fgm_cvar_data,
        csv_fname='LogReg/fits/fgm_cvar_fit_params.csv',
        plot_fname='LogReg/fits/fgm_cvar_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        log_scale=False,
    )

    plot_fits(
        dro_data = [
            LogReg_gd_cvar_data,
            LogReg_fgm_cvar_data,
        ],
        csv_files = [
            'LogReg/fits/gd_cvar_fit_params.csv',
            'LogReg/fits/fgm_cvar_fit_params.csv',
        ],
        plot_fname='LogReg/cvar_fit.pdf',
        plot_title=r'Logistic Regression: CVaR',
        log_scale=False,
    )



if __name__ == '__main__':
    main_logreg_exp()
    main_logreg_cvar()
