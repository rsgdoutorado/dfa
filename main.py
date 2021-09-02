import numpy as np
import dfa

if __name__ == '__main__':
    x = np.loadtxt('monofractal.txt')
    m = 2
    scale = [16, 32, 64, 100, 250, 500, 1000]
    q = np.linspace(-5, 5, 101)
    Fq, Hq, qRegLine, tq, hq, Dq = dfa.dfa_multifractal(x, m, scale, q)
