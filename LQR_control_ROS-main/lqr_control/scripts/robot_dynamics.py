#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint

def robot_dynamics(y, t, A, B, K):
    u = -np.dot(K, y)
    dydt = np.dot(A, y) + np.dot(B, u)
    return dydt
