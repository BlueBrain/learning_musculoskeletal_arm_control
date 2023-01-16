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

def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.04
    }

    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()

    p_num['m1'] = 0.2
    p_num['m2'] = 0.2
    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_ind):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
