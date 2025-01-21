"""
This script runs a Gillespie simulation of the homodimeric receptor model and
compares the mean and variance of these results to theoretical expressions.

Author: Duncan Kirby
Date: 2022-03-31
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from homodimeric import model as homodimeric_model
from pysb_methods import dose_response


# Homodimeric theory
def Homodimer(params, doses):
    Kb = params.k_b.value / params.kb.value
    Kt = params.k_t.value / params.kt.value
    Rt = params.R_0.value
    L = doses
    prefac = 1. / (8*Kb*L)
    Teq = Kt*(Kb+L)**2 + 4*Kb*L*Rt - (Kb+L)*np.sqrt(Kt*(Kt*(Kb+L)**2 + 8*Kb*L*Rt))
    return prefac * Teq


def HomodimerB(params, doses):
    Kb = params.k_b.value / params.kb.value
    Kt = params.k_t.value / params.kt.value
    Rt = params.R_0.value
    L = doses
    numerator = -Kb*Kt - Kt*L + np.sqrt(Kt * (Kt*(Kb + L)**2 + 8*Kb*L*Rt))
    return numerator / (4*Kb)


def homodimeric_var_theory(params, doses):
    Rt = params.R_0.value
    t = Homodimer(params, doses)
    b = HomodimerB(params, doses)
    # var = 1. / (2/t + 2/(Rt-t))
    var = 1. / (4. / (Rt-b-2*t) + 1. / t)
    # var = -(t*(b1-R1T+t)*(b2-R2T+t)) / (b2*R1T-R1T*R2T+b1*(R2T-b2)+t**2)
    return var


def homodimer_signal(params, doses, time):
    kpt = params.kp.value * time
    return kpt * Homodimer(params, doses)


def homodimeric_signal_var_theory(params, doses, time):
    kpt = params.kp.value * time
    meanR = Homodimer(params, doses)
    varR = homodimeric_var_theory(params, doses)
    var = kpt*meanR + kpt**2 * varR
    return var


if __name__ == '__main__':
    model_type = 'homodimeric'

    picorange = np.logspace(0, 6, 15)
    crange = picorange * 1E-12*1E-5*6.022E23
    t_end = 10000
    num_traj = 1000
    outdir = os.getcwd()

    # --------------------------------------------------------------------------
    params = homodimeric_model.parameters
    homodimeric_model.parameters['k_t'].value = homodimeric_model.parameters['k_b'].value
    doses = crange

    # theory
    theory_mean = Homodimer(params, doses)
    theory_signal_mean = homodimer_signal(params, doses, t_end)
    theory_var = homodimeric_var_theory(params, doses)
    theory_sig_var = homodimeric_signal_var_theory(homodimeric_model.parameters, doses, t_end)

    # simulation
    y = dose_response(homodimeric_model, doses, 'L_0', t_end, num_traj)
    mean_traj = np.mean(y['Cn'], axis=0)
    std_traj = np.std(y['Cn'], axis=0)
    mean_sig = np.mean(y['Nobs'], axis=0)
    std_sig = np.std(y['Nobs'], axis=0)

    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    for a in ax:
        a.spines['top'].set_color('none')
        a.spines['right'].set_color('none')
        a.set_xscale('log')
        a.set_xlabel('Ligand Concentration (pM)')

    cmap = ['#8acaff', '#1f69a6', '#7be082', '#25802b']

    # Validating the expressions for the mean
    ax[0].plot(picorange, mean_traj, c=cmap[0], label='receptor: simulation')
    ax[0].plot(picorange, theory_mean, c=cmap[1], linestyle='dashed', label='receptor: theory')
    ax[0].plot(picorange, mean_sig, c=cmap[2], label='signal: simulation')
    ax[0].plot(picorange, theory_signal_mean, c=cmap[3], linestyle='dashed', label='signal: theory')

    ax[0].set_ylabel('Mean')
    ax[0].legend(loc='upper left')

    # Validating the expressions for the variance
    ax[1].plot(picorange, std_traj, c=cmap[0], label='receptor: simulation')
    ax[1].plot(picorange, np.sqrt(theory_var), c=cmap[1], linestyle='dashed', label='receptor: theory')
    ax[1].plot(picorange, std_sig, c=cmap[2], label='signal: simulation')
    ax[1].plot(picorange, np.sqrt(theory_sig_var), c=cmap[3], linestyle='dashed', label='signal: theory')

    ax[1].set_ylabel('Standard Deviation')
    ax[1].legend(loc='upper left')

    plt.savefig(os.path.join(outdir, 'output', model_type + '_validation.pdf'))
    np.savetxt(os.path.join(outdir, 'output', "picorange.csv"), picorange, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "mean_sig.csv"), mean_sig, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "std_sig.csv"), std_sig, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "mean_rec.csv"), mean_traj, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "std_rec.csv"), std_traj, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "mean_theory_rec.csv"), theory_mean, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "std_theory_rec.csv"), np.sqrt(theory_var), delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "mean_theory_sig.csv"), theory_signal_mean, delimiter=",")
    np.savetxt(os.path.join(outdir, 'output', "std_theory_sig.csv"), np.sqrt(theory_sig_var), delimiter=",")
