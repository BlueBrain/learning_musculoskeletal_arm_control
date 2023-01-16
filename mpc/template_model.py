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


def template_model(obstacles):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Certain parameters
    m1 = 1.864572  # kg, mass of the first rod
    m2 = 1.534315  # kg, mass of the second rod
    L1 = 0.34  #m, length of the first rod
    L2 = 0.29  #m, length of the second rod
    l1 = L1/2
    l2 = L2/2
    J1 = (m1 * l1**2) / 3   # Inertia
    J2 = (m2 * l2**2) / 3   # Inertia

    m1 = model.set_variable('_p', 'm1')
    m2 = model.set_variable('_p', 'm2')

    g = -9.8066 # m/s^2, gravity

    h1 = (L1**2)*(0.25*m1+m2) + J1
    h2 = 0.5*m2*L2*L1
    h3 = L1*0.5*m2*L2
    h4 = -L1*g*(0.5*m1+m2)
    h5 = 0.5*L1*L2*m2
    h6 = 0.25*m2*(L2**2)+J2
    h7 = -0.5*m2*L2*L1
    h8 = -0.5*m2*L2*g

    # Setpoint x:
    pos_set = model.set_variable('_tvp', 'pos_set')

    # States struct (optimization variables):
    theta = model.set_variable('_x',  'theta', (2,1))
    dtheta = model.set_variable('_x',  'dtheta', (2,1))
    # Algebraic states:
    ddtheta = model.set_variable('_z', 'ddtheta', (2,1))

    # Input struct (optimization variables):
    u = model.set_variable('_u',  'force', (2,1))

    # Differential equations
    model.set_rhs('theta', dtheta)
    model.set_rhs('dtheta', ddtheta)

    # Euler Lagrange equations for the DIP system (in the form f(x,u,z) = 0)
    euler_lagrange = vertcat(
        # 1
        h1*ddtheta[0] + h2*ddtheta[1]*cos(theta[0]-theta[1])
        +h3*dtheta[1]**2*sin(theta[0]-theta[1]) + h4*sin(theta[0]) + u[0],
        # 2
        h5*cos(theta[0]-theta[1])*ddtheta[0] + h6*ddtheta[1] + h7*sin(theta[0]-theta[1])*dtheta[0]**2
        + h8*sin(theta[1]) + u[1]
    )

    model.set_alg('euler_lagrange', euler_lagrange)

    # Expressions for kinetic and potential energy

    E_kin = ((1./8.)*m1*L1**2)*dtheta[0]**2 + 0.5*m2*((L1**2)*(dtheta[0]**2) + 0.25*(L2**2)*(dtheta[1]**2) + L1*L2*dtheta[0]*dtheta[1]*cos(theta[0]-theta[1])) + 0.5*J1*dtheta[0]**2 + 0.5*J2*dtheta[1]**2
    E_pot = 0.5*m1*g*L1*cos(theta[0]) + m2*g*(L1*cos(theta[0]) + 0.5*L2*cos(theta[1]))

    model.set_expression('E_kin', E_kin)
    model.set_expression('E_pot', E_pot)

    # Coordinates of the nodes:
    node1_x = L1*sin(model.x['theta',0])
    node1_y = -L1*cos(model.x['theta',0])

    node2_x = node1_x+L2*sin(model.x['theta',1])
    node2_y = node1_y-L2*cos(model.x['theta',1])

    # Calculations to avoid obstacles:
    obstacle_distance = []

    for obs in obstacles:
        d0 = sqrt((node1_x-obs['x'])**2+(node1_y-obs['y'])**2)-obs['r']*1.25
        d1 = sqrt((node2_x-obs['x'])**2+(node2_y-obs['y'])**2)-obs['r']*1.25
        obstacle_distance.extend([d0, d1])

    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))
    model.set_expression('tvp', pos_set)

    # Build the model
    model.setup()

    return model
