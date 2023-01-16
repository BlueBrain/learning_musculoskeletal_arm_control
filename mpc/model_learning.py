#!/usr/bin/env python

"""model_learning.py: Source code of the model predictive control to obtain optimum trajectories for 2D arm movement

This module demonstrates how to use a do-mpc module to generate optimum target trajectories

Example:
    You can directly execute with python command and pass -r [Trial Name] to name the execution, see below for other arguments ::

        $ python model_learning.py --show_animation True --store_animation True --store_results True -r fixed_theta0

It saves the trajectories obtained from the optimum solution while adjusting the timestep for musculoskeletal control

Options:
  --show_animation             Visualize the animation
  --store_animation            Save the animation as a gif
  --store_results              Save the mpc results along with joint angles to be used in musculoskeletal control
  -r        --run              Name the execution to use in output images and files to be recorded
"""

__author__ = "Berat Denizdurduran"
__copyright__ = "Copyright 2022, Berat Denizdurduran"
__license__ = "public, published"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "After-publication"

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
#sys.path.append('../../')
import do_mpc
from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# Plot settings
rcParams['text.usetex'] = False
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

import time
import argparse

from template_mpc import template_mpc
from template_simulator import template_simulator
from template_model import template_model


""" User settings: """
parser = argparse.ArgumentParser()
parser.add_argument("--show_animation", help="Visualize the animation", default=True)
parser.add_argument("--store_animation", help="Save the animation", default=True)
parser.add_argument("--store_results", help="Save the mpc results", default=True)
parser.add_argument("-r", "--run", required=True, help="Run name")
args = parser.parse_args()

show_animation = True
store_animation = True
store_results = True

# Define obstacles to avoid (cicles)
# here there are two obtstacles at the close proximity of the arm to help mpc to find the best trajectories
obstacles = [
    {'x': 0.25, 'y': -0.225, 'r': 0.1},
    {'x': -0.12, 'y': -0.225, 'r': 0.1},
]

scenario = 1  # 1 = down-down start, 2 = up-up start, both with setpoint change.

"""
Get configured do-mpc modules:
"""

model = template_model(obstacles)
simulator = template_simulator(model)
mpc = template_mpc(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

if scenario == 1:
    simulator.x0['theta'] = 0.
elif scenario == 2:
    simulator.x0['theta'] = np.pi
else:
    raise Exception('Scenario not defined.')

x0 = simulator.x0.cat.full()

mpc.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Setup graphic:
"""

# Function to create lines:
L1 = 0.34  #m, length of the first rod
L2 = 0.29  #m, length of the second rod
def pendulum_bars(x):
    x = x.flatten()
    # Get the x,y coordinates of the two bars for the given state x.
    line_1_x = np.array([
        0,
        L1*np.sin(x[0])
    ])

    line_1_y = np.array([
        0,
        -L1*np.cos(x[0])
    ])

    line_2_x = np.array([
        line_1_x[1],
        line_1_x[1] + L2*np.sin(x[1])
    ])

    line_2_y = np.array([
        line_1_y[1],
        line_1_y[1] - L2*np.cos(x[1])
    ])

    line_1 = np.stack((line_1_x, line_1_y))
    line_2 = np.stack((line_2_x, line_2_y))

    return line_1, line_2

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

fig = plt.figure(figsize=(18,10))
plt.ion()

ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((1, 2), (0, 1))

ax2.set_ylabel('Relative Angle (Radian)')
ax2.set_xlabel('Time (Second)')

mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax2)

ax1.axhline(0,color='black')

# Axis on the right.
for ax in [ax2]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

bar1 = ax1.plot([],[], '-o', linewidth=5, markersize=10)
bar2 = ax1.plot([],[], '-o', linewidth=5, markersize=10)

"""
# uncomment this to visualize the obstacles
for obs in obstacles:
    circle = Circle((obs['x'], obs['y']), obs['r'], edgecolor='red',facecolor='red')
    ax1.add_artist(circle)
"""

ax1.set_xlim(-0.8,0.8)
ax1.set_ylim(-0.8,0.8)
ax1.set_axis_off()

fig.align_ylabels()
fig.tight_layout()


"""
Run MPC main loop:
"""
time_list = []

n_steps = 40
for k in range(n_steps):
    tic = time.time()
    u0 = mpc.make_step(x0)
    toc = time.time()
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    time_list.append(toc-tic)

    if args.show_animation:
        line1, line2 = pendulum_bars(x0)
        bar1[0].set_data(line1[0],line1[1])
        bar2[0].set_data(line2[0],line2[1])
        mpc_graphics.plot_results()
        mpc_graphics.plot_predictions()
        mpc_graphics.reset_axes()
        plt.show()
        plt.savefig("animation/{}_{}".format(args.run, k))
        plt.pause(0.04)

time_arr = np.array(time_list)
mean = np.round(np.mean(time_arr[1:])*1000)
var = np.round(np.std(time_arr[1:])*1000)
print('mean runtime:{}ms +- {}ms for MPC step'.format(mean, var))

# The function describing the gif:
if args.store_animation:
    x_arr = mpc.data['_x']
    def update(t_ind):
        line1, line2 = pendulum_bars(x_arr[t_ind])
        bar1[0].set_data(line1[0],line1[1])
        bar2[0].set_data(line2[0],line2[1])
        mpc_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=20)
    anim.save('animation/{}.gif'.format(args.run), writer=gif_writer)

# Store results:
if args.store_results:
    do_mpc.data.save_results([mpc, simulator], args.run)
    # change of dt to adjust the trajectory for opensim-rl timesteps
    t_mpc = np.arange(0,1.6,0.04)
    f_elbow = interpolate.interp1d(t_mpc, mpc.data['_x'][0:,1])
    f_shoulder = interpolate.interp1d(t_mpc, mpc.data['_x'][0:,0])
    f_elbow_vel = interpolate.interp1d(t_mpc, mpc.data['_x'][0:,3])
    f_shoulder_vel = interpolate.interp1d(t_mpc, mpc.data['_x'][0:,2])
    f_elbow_acc = interpolate.interp1d(t_mpc, mpc.data['_z'][0:,1])
    f_shoulder_acc = interpolate.interp1d(t_mpc, mpc.data['_z'][0:,0])
    t_arm = np.arange(0, 0.75, 0.01)
    target_elbow = f_elbow(t_arm)
    target_shoulder = f_shoulder(t_arm)
    target_elbow_vel = f_elbow_vel(t_arm)
    target_shoulder_vel = f_shoulder_vel(t_arm)
    target_elbow_acc = f_elbow_acc(t_arm)
    target_shoulder_acc = f_shoulder_acc(t_arm)
    np.save("results/target_of_elbow_{}".format(args.run), target_elbow)
    np.save("results/target_of_shoulder_{}".format(args.run), target_shoulder)
    np.save("results/target_of_elbow_vel_{}".format(args.run), target_elbow_vel)
    np.save("results/target_of_shoulder_vel_{}".format(args.run), target_shoulder_vel)
    np.save("results/target_of_elbow_acc_{}".format(args.run), target_elbow_acc)
    np.save("results/target_of_shoulder_acc_{}".format(args.run), target_shoulder_acc)

input('Learning finished! Press any key to exit.')
