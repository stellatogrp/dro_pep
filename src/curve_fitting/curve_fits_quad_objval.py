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
    # ax.set_title(rf'$\rho = {rho}, \gamma = {gamma}, C = {C}$')
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


def main_QP_nonstrong_nonstrcvxFGM_exp():
    QP_nonstrong_gd_exp = pd.read_csv(f'Quad_NonstrongCvx_obj_val/dro/grad_desc/dro.csv')
    QP_nonstrong_gd_exp_data = QP_nonstrong_gd_exp[QP_nonstrong_gd_exp['eps_idx'] == 4]
    QP_nonstrong_gd_pep = pd.read_csv('Quad_NonstrongCvx_obj_val/dro/grad_desc/pep.csv')
    QP_nonstrong_gd_pep_data = QP_nonstrong_gd_pep[QP_nonstrong_gd_pep['obj'] == 'obj_val']['val'].to_numpy()

    QP_nonstrong_ngd_exp = pd.read_csv(f'Quad_NonstrongCvx_obj_val/dro/nesterov_grad_desc/dro.csv')
    QP_nonstrong_ngd_exp_data = QP_nonstrong_ngd_exp[QP_nonstrong_ngd_exp['eps_idx'] == 4]
    QP_nonstrong_ngd_pep = pd.read_csv('Quad_NonstrongCvx_obj_val/dro/nesterov_grad_desc/pep.csv')
    QP_nonstrong_ngd_pep_data = QP_nonstrong_ngd_pep[QP_nonstrong_ngd_pep['obj'] == 'obj_val']['val'].to_numpy()

    curve_fit(
        QP_nonstrong_gd_exp_data,
        csv_fname='Quad_NonstrongCvx_obj_val/fits/gd_exp_fit_params.csv',
        plot_fname='Quad_NonstrongCvx_obj_val/fits/gd_exp_fit_plot.pdf',
        gamma_fix=False,
        eta_fix=False,
    )

    curve_fit(
        QP_nonstrong_gd_pep_data,
        csv_fname='Quad_NonstrongCvx_obj_val/fits/gd_pep_fit_params.csv',
        plot_fname='Quad_NonstrongCvx_obj_val/fits/gd_pep_fit_plot.pdf',
        gamma_fix=False,
        eta_fix=False,
    )
    
    curve_fit(
        QP_nonstrong_ngd_exp_data,
        csv_fname='Quad_NonstrongCvx_obj_val/fits/ngd_exp_fit_params.csv',
        plot_fname='Quad_NonstrongCvx_obj_val/fits/ngd_exp_fit_plot.pdf',
        gamma_fix=False,
        eta_fix=False,
    )

    curve_fit(
        QP_nonstrong_ngd_pep_data,
        csv_fname='Quad_NonstrongCvx_obj_val/fits/ngd_pep_fit_params.csv',
        plot_fname='Quad_NonstrongCvx_obj_val/fits/ngd_pep_fit_plot.pdf',
        gamma_fix=False,
        eta_fix=False,
    )

    plot_fits(
        dro_data = [
            QP_nonstrong_gd_pep_data,
            QP_nonstrong_ngd_pep_data,
            QP_nonstrong_gd_exp_data,
            QP_nonstrong_ngd_exp_data,
        ],
        csv_files = [
            'Quad_NonstrongCvx_obj_val/fits/gd_pep_fit_params.csv',
            'Quad_NonstrongCvx_obj_val/fits/ngd_pep_fit_params.csv',
            'Quad_NonstrongCvx_obj_val/fits/gd_exp_fit_params.csv',
            'Quad_NonstrongCvx_obj_val/fits/ngd_exp_fit_params.csv',
        ],
        plot_fname = 'Quad_NonstrongCvx_obj_val/exp_fit.pdf',
        plot_title = r'Quadratic minimization ($\mu=0$) with non-strcvx FGM: Expectation',
    )


def main_QP_strong_nonstrcvxFGM_exp():
    # QP_strong_gd_exp = pd.read_csv(f'Quad_StrongCvx_obj_val/dro/grad_desc/dro.csv')
    # QP_strong_gd_exp_data = QP_strong_gd_exp[QP_strong_gd_exp['eps_idx'] == 3]
    QP_strong_gd_pep = pd.read_csv('Quad_StrongCvx_obj_val/dro/grad_desc/pep.csv')
    QP_strong_gd_pep_data = QP_strong_gd_pep[QP_strong_gd_pep['obj'] == 'obj_val']['val'].to_numpy()

    # QP_strong_ngd_exp = pd.read_csv(f'Quad_StrongCvx_obj_val/dro/nesterov_grad_desc/dro.csv')
    # QP_strong_ngd_exp_data = QP_strong_ngd_exp[QP_strong_ngd_exp['eps_idx'] == 3]
    QP_strong_ngd_pep = pd.read_csv('Quad_StrongCvx_obj_val/dro/nesterov_grad_desc/pep.csv')
    QP_strong_ngd_pep_data = QP_strong_ngd_pep[QP_strong_ngd_pep['obj'] == 'obj_val']['val'].to_numpy()

    # curve_fit(
    #     QP_strong_gd_exp_data,
    #     csv_fname='Quad_StrongCvx_obj_val/fits/gd_exp_fit_params.csv',
    #     plot_fname='Quad_StrongCvx_obj_val/fits/gd_exp_log_fit_plot.pdf',
    #     gamma_fix=False,
    #     gamma_val=0.0,
    #     rho_fix=True,
    #     rho_val=(1.0-1./25)**2,
    #     eta_fix=True,
    #     eta_val=0.0,
    #     log_scale=False,
    # )

    curve_fit(
        QP_strong_gd_pep_data,
        csv_fname='Quad_StrongCvx_obj_val/fits/gd_pep_fit_params.csv',
        plot_fname='Quad_StrongCvx_obj_val/fits/gd_pep_log_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        eta_fix=False,
        log_scale=False,
    )
    
    # curve_fit(
    #     QP_strong_ngd_exp_data,
    #     csv_fname='Quad_StrongCvx_obj_val/fits/ngd_exp_fit_params.csv',
    #     plot_fname='Quad_StrongCvx_obj_val/fits/ngd_exp_fit_plot.pdf',
    #     gamma_fix=True,
    #     gamma_val=0.5,
    #     rho_fix=False,
    #     rho_val = (np.sqrt(25.) - 1.) / (np.sqrt(25.) + 1.) * (1 - 1./25),
    #     eta_fix=True,
    #     eta_val=0.0,
    #     log_scale=False,
    # )

    curve_fit(
        QP_strong_ngd_pep_data,
        csv_fname='Quad_StrongCvx_obj_val/fits/ngd_pep_fit_params.csv',
        plot_fname='Quad_StrongCvx_obj_val/fits/ngd_pep_log_fit_plot.pdf',
        rho_fix=False,
        gamma_fix=False,
        eta_fix=False,
        log_scale=False,
    )

    plot_fits(
        dro_data = [
            # QP_strong_gd_exp_data,
            # QP_strong_ngd_exp_data,
            QP_strong_gd_pep_data,
            QP_strong_ngd_pep_data,
        ],
        csv_files = [
            # 'Quad_StrongCvx_obj_val/fits/gd_exp_fit_params.csv',
            # 'Quad_StrongCvx_obj_val/fits/ngd_exp_fit_params.csv',
            'Quad_StrongCvx_obj_val/fits/gd_pep_fit_params.csv',
            'Quad_StrongCvx_obj_val/fits/ngd_pep_fit_params.csv',
        ],
        plot_fname = 'Quad_StrongCvx_obj_val/pep_fit.pdf',
        plot_title = r'Quadratic minimization ($\mu>0$) with non-strcvx FGM: PEP',
        log_scale=False,
    )




if __name__ == '__main__':
    main_QP_nonstrong_nonstrcvxFGM_exp()
    main_QP_strong_nonstrcvxFGM_exp()