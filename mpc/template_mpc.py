__author__ = "Berat Denizdurduran"
__copyright__ = "Copyright 2022, Berat Denizdurduran"
__license__ = "public, published"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "After-publication"

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 40,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.04,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'ma27'}
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['E_kin'] - model.aux['E_pot'] + (model.x['theta',1] - model.tvp['pos_set'])**2
    lterm = 1000*(model.x['theta',0])**2 + (model.x['theta',1] - np.pi/2)**2 + model.u['force',1]**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(force=0.01)

    mpc.bounds['lower','_u','force'] = -0.35
    mpc.bounds['upper','_u','force'] = 0.35

    # Avoid the obstacles:
    mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)

    # Values for the masses (for robust MPC)
    m1_var = 0.2*np.array([1, 0.95, 1.05])
    m2_var = 0.2*np.array([1, 0.95, 1.05])
    mpc.set_uncertainty_values(m1=m1_var, m2=m2_var)

    tvp_template = mpc.get_tvp_template()

    # When to switch setpoint:
    t_switch = 1.0   # seconds
    ind_switch = t_switch // setup_mpc['t_step']

    def tvp_fun(t_ind):
        ind = t_ind // setup_mpc['t_step']
        if ind <= ind_switch:
            tvp_template['_tvp',:, 'pos_set'] = np.pi/2
        else:
            tvp_template['_tvp',:, 'pos_set'] = np.pi/2
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc
