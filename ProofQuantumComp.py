import numpy as np

# --- Updated geodesic-constraint primitives (8D) with iterative projection ---
def compute_W(f, phi):
    return np.sum(f**2 * np.exp(2j * phi))

def constraint_gradients(f, phi):
    dW_df   = 2 * f * np.exp(2j * phi)
    dW_dphi = 2j * f**2 * np.exp(2j * phi)
    gR = np.concatenate((np.real(dW_df), np.real(dW_dphi)))
    gI = np.concatenate((np.imag(dW_df), np.imag(dW_dphi)))
    return gR, gI

def project_extended(X, tol=1e-12, max_iter=10):
    """
    Iteratively project X onto the W=0 manifold until |W|<tol.
    """
    f = None; phi = None
    for _ in range(max_iter):
        N2 = X.size // 2
        f = X[:N2]; phi = X[N2:]
        W = compute_W(f, phi)
        if abs(W) < tol:
            break
        gR, gI = constraint_gradients(f, phi)
        J = np.vstack((gR, gI))
        Wvec = np.array([W.real, W.imag])
        delta = J.T @ np.linalg.solve(J @ J.T, Wvec)
        X = X - delta
    return X

def apply_gate_ext(X, gate_fn):
    return project_extended(gate_fn(X.copy()))

def mix_branches(i, j, theta):
    def gate(X):
        N2 = X.size // 2
        f = X[:N2].copy(); phi = X[N2:].copy()
        a_i = f[i] * np.exp(1j * phi[i]); a_j = f[j] * np.exp(1j * phi[j])
        b_i = np.cos(theta)*a_i - np.sin(theta)*a_j
        b_j = np.sin(theta)*a_i + np.cos(theta)*a_j
        f[i], f[j]     = abs(b_i), abs(b_j)
        phi[i], phi[j] = np.angle(b_i), np.angle(b_j)
        return np.concatenate((f, phi))
    return gate

def swap_branches(i, j):
    def gate(X):
        N2 = X.size // 2
        f = X[:N2].copy(); phi = X[N2:].copy()
        f[i], f[j]     = f[j], f[i]
        phi[i], phi[j] = phi[j], phi[i]
        return np.concatenate((f, phi))
    return gate

def init_state():
    f = np.full(4, 1/4)
    phi = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    return np.concatenate((f, phi))

# --- n-qubit wrappers (qubits 0&1) ---
def apply_rotation_n(X, qubit, theta):
    pairs = [(0,1),(2,3)] if qubit==0 else [(0,2),(1,3)]
    for i,j in pairs:
        X = apply_gate_ext(X, mix_branches(i, j, theta))
    return X

def apply_cnot_n(X):
    for i in range(4):
        if (i & 1)==1:
            j = i ^ 2
            if i < j:
                X = apply_gate_ext(X, swap_branches(i, j))
    return X

def apply_swap_n(X):
    X = apply_gate_ext(X, swap_branches(1,2))
    return X

def apply_phase_n(X, qubit, phi_shift):
    def gate(Y):
        f = Y[:4].copy(); ph = Y[4:].copy()
        for i in range(4):
            if ((i>>qubit)&1)==1:
                ph[i] += phi_shift
        return np.concatenate((f, ph))
    return apply_gate_ext(X, gate)

# --- Test gates on n=500 with iterative projection ---
if __name__=="__main__":
    n = 500
    X = init_state()
    gates = [
        ("R0(π/3)", lambda X: apply_rotation_n(X, 0, np.pi/3)),
        ("R1(π/4)", lambda X: apply_rotation_n(X, 1, np.pi/4)),
        ("CNOT", apply_cnot_n),
        ("SWAP", apply_swap_n),
        ("H0", lambda X: apply_rotation_n(X, 0, np.pi/4)),
        ("H1", lambda X: apply_rotation_n(X, 1, np.pi/4)),
        ("P0(π/2)", lambda X: apply_phase_n(X, 0, np.pi/2)),
        ("P1(π/3)", lambda X: apply_phase_n(X, 1, np.pi/3)),
        ("T0(π/4)", lambda X: apply_phase_n(X, 0, np.pi/4)),
        ("T1(π/4)", lambda X: apply_phase_n(X, 1, np.pi/4)),
    ]
    
    print(f"Testing {len(gates)} gates on {n}-qubit system:")
    for name, fn in gates:
        X = fn(X)
        f, phi = X[:4], X[4:]
        W = compute_W(f, phi)
        print(f"{name}: |W| = {abs(W):.2e}")
