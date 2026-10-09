"""Project 3: the Schrodinger equation, built, solved, tested and rediscovered.

Units are atomic-style throughout: hbar = m = 1, so the equation reads

    i d(psi)/dt = -1/2 d2(psi)/dx2 + V(x) psi.

    grid        the spatial grid, the momentum grid and the discrete Hamiltonian
    stationary  eigenstates, and the closed forms they are checked against
    evolve      split-operator time evolution and the observables it conserves
    born        what psi predicts about where a particle is found
    tunneling   a wave packet against a barrier, against the plane-wave formula
    discover    the equation read back out of simulated psi(x, t)
    symbolic    closed forms found by symbolic regression (PySR when installed)
"""
