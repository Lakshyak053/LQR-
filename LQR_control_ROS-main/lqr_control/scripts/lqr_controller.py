#!/usr/bin/env python3

import numpy as np
from scipy.linalg import solve_continuous_are

def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    return K
