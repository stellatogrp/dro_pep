import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sublinear_curve_fit(data, csv_fname=None, plot_fname=None, mincutoff=10):
    K = data['K'].to_numpy()
    phi = data['dro_feas_sol'].to_numpy()
    A = K.reshape(-1, 1)
    A = np.hstack([np.log(A), np.ones(A.shape)])
    b = np.log(phi)

    A = A[mincutoff-1:, :]
    b = b[mincutoff-1:]

    b_inv = np.abs(1/b)

    x = cp.Variable(A.shape[1])
    obj = .5 * cp.sum_squares(cp.multiply(b_inv, A @ x - b))
    constraints = []

    constraints += [A @ x >= b]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    res = prob.solve()
    print(x.value)

    gamma = -x.value[0]
    C = np.exp(x.value[1])

    fitted_vals = C * np.pow(K, -gamma)
    print(fitted_vals)

    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(K, phi, label='true vals')
    ax.plot(K, fitted_vals, label='fitted curve')
    ax.set_title(rf'$\gamma = {gamma}, C = {C}$')
    ax.grid(color='lightgray', alpha=0.3)
    ax.legend()
    # plt.show()
    if csv_fname is not None:
        data = [
            {'gamma': gamma, 'C': C},
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_fname, index=False)
    if plot_fname is not None:
        plt.savefig(plot_fname)


def linear_curve_fit(data, csv_fname=None, plot_fname=None, mincutoff=10):
    K = data['K'].to_numpy()
    phi = data['dro_feas_sol'].to_numpy()
    A = K.reshape(-1, 1)
    A = np.hstack([A, np.ones(A.shape)])
    b = np.log(phi)
    A = A[mincutoff-1:, :]
    b = b[mincutoff-1:]

    b_inv = np.abs(1/b)

    x = cp.Variable(A.shape[1])
    obj = .5 * cp.sum_squares(cp.multiply(b_inv, A @ x - b))
    constraints = []

    constraints += [A @ x >= b]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    res = prob.solve()
    print(x.value)

    gamma = np.exp(x.value[0])
    C = np.exp(x.value[1])

    fitted_vals = C * np.pow(gamma, K)
    print(fitted_vals)

    fig, ax = plt.subplots()
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(K, phi, label='true vals')
    ax.plot(K, fitted_vals, label='fitted curve')
    ax.set_title(rf'$\gamma = {gamma}, C = {C}$')
    ax.grid(color='lightgray', alpha=0.3)
    ax.legend()
    # plt.show()
    if csv_fname is not None:
        data = [
            {'gamma': gamma, 'C': C},
        ]
        df = pd.DataFrame(data)
        df.to_csv(csv_fname, index=False)
    if plot_fname is not None:
        plt.savefig(plot_fname)


def main_QP_nonstrong_exp():
    QP_nonstrong_gd_exp = pd.read_csv(f'Quad_NonstrongCvx/dro/grad_desc_exp_1_30/dro.csv')
    QP_nonstrong_gd_exp_data = QP_nonstrong_gd_exp[QP_nonstrong_gd_exp['eps_idx'] == 5]
    QP_nonstrong_ngd_exp = pd.read_csv(f'Quad_NonstrongCvx/dro/nesterov_grad_desc_exp_1_30/dro.csv')
    QP_nonstrong_ngd_exp_data = QP_nonstrong_ngd_exp[QP_nonstrong_ngd_exp['eps_idx'] == 5]

    sublinear_curve_fit(
        QP_nonstrong_gd_exp_data,
        csv_fname='Quad_NonstrongCvx/gd_exp_fit_params.csv',
        plot_fname='Quad_NonstrongCvx/gd_exp_fit_plot.pdf',
    )

    sublinear_curve_fit(
        QP_nonstrong_ngd_exp_data,
        csv_fname='Quad_NonstrongCvx/ngd_exp_fit_params.csv',
        plot_fname='Quad_NonstrongCvx/ngd_exp_fit_plot.pdf',
    )

def main_QP_nonstrong_cvar():
    QP_nonstrong_gd_cvar = pd.read_csv(f'Quad_NonstrongCvx/dro/grad_desc_cvar_1_30/dro.csv')
    QP_nonstrong_gd_cvar_data = QP_nonstrong_gd_cvar[QP_nonstrong_gd_cvar['eps_idx'] == 3]
    QP_nonstrong_ngd_cvar = pd.read_csv(f'Quad_NonstrongCvx/dro/nesterov_grad_desc_cvar_1_30/dro.csv')
    QP_nonstrong_ngd_cvar_data = QP_nonstrong_ngd_cvar[QP_nonstrong_ngd_cvar['eps_idx'] == 3]

    sublinear_curve_fit(
        QP_nonstrong_gd_cvar_data,
        csv_fname='Quad_NonstrongCvx/gd_cvar_fit_params.csv',
        plot_fname='Quad_NonstrongCvx/gd_cvar_fit_plot.pdf',
    )

    sublinear_curve_fit(
        QP_nonstrong_ngd_cvar_data,
        csv_fname='Quad_NonstrongCvx/ngd_cvar_fit_params.csv',
        plot_fname='Quad_NonstrongCvx/ngd_cvar_fit_plot.pdf',
    )

def main_QP_strong_exp():
    QP_strong_gd_exp = pd.read_csv(f'Quad_StrongCvx/dro/grad_desc_exp_1_30/dro.csv')
    QP_strong_gd_exp_data = QP_strong_gd_exp[QP_strong_gd_exp['eps_idx'] == 4]
    QP_strong_ngd_exp = pd.read_csv(f'Quad_StrongCvx/dro/nesterov_grad_desc_exp_1_30/dro.csv')
    QP_strong_ngd_exp_data = QP_strong_ngd_exp[QP_strong_ngd_exp['eps_idx'] == 4]

    linear_curve_fit(
        QP_strong_gd_exp_data,
        csv_fname='Quad_StrongCvx/gd_exp_fit_params.csv',
        plot_fname='Quad_StrongCvx/gd_exp_fit_plot.pdf',
    )

    linear_curve_fit(
        QP_strong_ngd_exp_data,
        csv_fname='Quad_StrongCvx/ngd_exp_fit_params.csv',
        plot_fname='Quad_StrongCvx/ngd_exp_fit_plot.pdf',
    )

def main_QP_strong_cvar():
    QP_strong_gd_cvar = pd.read_csv(f'Quad_StrongCvx/dro/grad_desc_cvar_1_30/dro.csv')
    QP_strong_gd_cvar_data = QP_strong_gd_cvar[QP_strong_gd_cvar['eps_idx'] == 4]
    QP_strong_ngd_cvar = pd.read_csv(f'Quad_StrongCvx/dro/nesterov_grad_desc_cvar_1_30/dro.csv')
    QP_strong_ngd_cvar_data = QP_strong_ngd_cvar[QP_strong_ngd_cvar['eps_idx'] == 4]

    linear_curve_fit(
        QP_strong_gd_cvar_data,
        csv_fname='Quad_StrongCvx/gd_cvar_fit_params.csv',
        plot_fname='Quad_StrongCvx/gd_cvar_fit_plot.pdf',
    )

    linear_curve_fit(
        QP_strong_ngd_cvar_data,
        csv_fname='Quad_StrongCvx/ngd_cvar_fit_params.csv',
        plot_fname='Quad_StrongCvx/ngd_cvar_fit_plot.pdf',
    )

def main_logreg_exp():
    LogReg_gd_exp = pd.read_csv(f'LogReg/dro/grad_desc_exp_1_30/dro.csv')
    LogReg_gd_exp_data = LogReg_gd_exp[LogReg_gd_exp['eps_idx'] == 5]
    LogReg_ngd_exp = pd.read_csv(f'LogReg/dro/nesterov_grad_desc_exp_1_30/dro.csv')
    LogReg_ngd_exp_data = LogReg_ngd_exp[LogReg_ngd_exp['eps_idx'] == 5]

    linear_curve_fit(
        LogReg_gd_exp_data,
        csv_fname='LogReg/gd_exp_fit_params.csv',
        plot_fname='LogReg/gd_exp_fit_plot.pdf',
        # mincutoff=5,
    )

    linear_curve_fit(
        LogReg_ngd_exp_data,
        csv_fname='LogReg/ngd_exp_fit_params.csv',
        plot_fname='LogReg/ngd_exp_fit_plot.pdf',
    )

def main_logreg_cvar():
    LogReg_gd_cvar = pd.read_csv(f'LogReg/dro/grad_desc_cvar_1_30/dro.csv')
    LogReg_gd_cvar_data = LogReg_gd_cvar[LogReg_gd_cvar['eps_idx'] == 3]
    LogReg_ngd_cvar = pd.read_csv(f'LogReg/dro/nesterov_grad_desc_cvar_1_30/dro.csv')
    LogReg_ngd_cvar_data = LogReg_ngd_cvar[LogReg_ngd_cvar['eps_idx'] == 2]

    linear_curve_fit(
        LogReg_gd_cvar_data,
        csv_fname='LogReg/gd_cvar_fit_params.csv',
        plot_fname='LogReg/gd_cvar_fit_plot.pdf',
        # mincutoff=5,
    )

    linear_curve_fit(
        LogReg_ngd_cvar_data,
        csv_fname='LogReg/ngd_cvar_fit_params.csv',
        plot_fname='LogReg/ngd_cvar_fit_plot.pdf',
    )


def main_lasso_exp():
    Lasso_ista_exp = pd.read_csv(f'Lasso/dro/ISTA_exp_1_25/dro.csv')
    Lasso_ista_exp_data = Lasso_ista_exp[Lasso_ista_exp['eps_idx'] == 2]
    Lasso_fista_exp = pd.read_csv(f'Lasso/dro/FISTA_exp_1_25/dro.csv')
    Lasso_fista_exp_data = Lasso_fista_exp[Lasso_fista_exp['eps_idx'] == 2]

    sublinear_curve_fit(
        Lasso_ista_exp_data,
        csv_fname='Lasso/ista_exp_fit_params.csv',
        plot_fname='Lasso/ista_exp_fit_plot.pdf',
        mincutoff=1,
    )

    linear_curve_fit(
        Lasso_fista_exp_data,
        csv_fname='Lasso/fista_exp_fit_params.csv',
        plot_fname='Lasso/fista_exp_fit_plot.pdf',
        mincutoff=15,
    )

def main_lasso_cvar():
    Lasso_ista_cvar = pd.read_csv(f'Lasso/dro/ISTA_exp_1_25/dro.csv')
    Lasso_ista_cvar_data = Lasso_ista_exp[Lasso_ista_exp['eps_idx'] == 2]
    Lasso_fista_exp = pd.read_csv(f'Lasso/dro/FISTA_exp_1_25/dro.csv')
    Lasso_fista_exp_data = Lasso_fista_exp[Lasso_fista_exp['eps_idx'] == 2]

    sublinear_curve_fit(
        Lasso_ista_exp_data,
        csv_fname='Lasso/ista_exp_fit_params.csv',
        plot_fname='Lasso/ista_exp_fit_plot.pdf',
        mincutoff=10,
    )

    linear_curve_fit(
        Lasso_fista_exp_data,
        csv_fname='Lasso/fista_exp_fit_params.csv',
        plot_fname='Lasso/fista_exp_fit_plot.pdf',
        mincutoff=15,
    )


if __name__ == '__main__':
    # main_QP_nonstrong_exp()
    # main_QP_nonstrong_cvar()
    # main_QP_strong_exp()
    # main_QP_strong_cvar()
    # main_logreg_exp()
    # main_logreg_cvar()
    main_lasso_exp()
