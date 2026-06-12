import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# TODO: refactor: move W and h, ensure immutability of the parameters
 
# Connectivity matrix
# W[i, j] = weight from population j → population i
#
# Derived from Pfeffer et al. 2013 + Litwin-Kumar et al. 2016
# Additionally scaled so that for all eigenvalues: 0 < eign < 1
#   PV  → E, PV (strong self-inhibition)
#   SOM → E, PV, VIP (avoids SOM→SOM)
#   VIP → SOM (strong), PV (weak)
#   E   → PV, SOM, VIP (broad excitatory drive)
 
W = np.array([
    #  E      PV     SOM    VIP     ← presynaptic
    [ 0.6,  -0.5,  -0.4,   0.0 ],  # E   (post)
    [ 0.7,  -0.2,  -0.2,  -0.1 ],  # PV  (post)
    [ 0.3,   0.0,   0.0,  -0.6 ],  # SOM (post)
    [ 0.2,  -0.1,  -0.6,   0.0 ],  # VIP (post)
])
 
# Neuron parameters
 
POP_NAMES = ["E", "PV", "SOM", "VIP"]
TAU = np.array([20.0, 10.0, 25.0, 20.0]) # Time constants
R_TARGET = np.array([8.0, 12.0, 5.0, 4.0]) # Firing rate targets

#

h = (np.eye(len(W[0,:])) - W) @ R_TARGET
 
# Transfer function
 
def sigmoid(x, r_max, threshold, gain):
    """Rectified sigmoid transfer function (firing rate nonlinearity)."""
    return r_max / (1.0 + np.exp(-gain * (x - threshold)))

def relu(x):
    """Threshold-linear transfer function (ReLu)."""
    return np.maximum(0.0, x)


# Fixed point and stability

def find_fp(W, h, r0=None):
    if r0 is None:
        r0 = np.abs(np.linalg.solve(np.eye(len(W[0,:])) - W, h))

    def residual(r):
        return r - relu(W @ r + h)
    
    r_star = fsolve(residual, r0)
    return relu(r_star)

def jacobian(W, r_star):
    D = np.diag((r_star > 1e-9).astype(float))
    return np.diag(1.0 / TAU) @ (D @ W - np.eye(len(W[0,:])))

def check_W(W):
    """Report spectral radius and whether stable FP is guaranteed."""
    eigs = np.linalg.eigvals(W)
    sr   = np.max(np.real(eigs))
    return sr, sr < 1.0

def fp_report(W, h):
    sr, ok = check_W(W)
    r_star  = find_fp(W, h)
    J       = jacobian(W, r_star)
    jeigs   = np.linalg.eigvals(J)
    stable  = np.all(np.real(jeigs) < 0)
    h_ok    = np.all(h > 0)

    print(f"W spectral radius:  {sr:.4f}  {'< 1' if ok else '>= 1 — FP may not exist!'}")
    print(f"h all positive:     {'yes' if h_ok else 'NO! — reduce inhibitory weights or lower r_target'}")
    print(f"Jacobian stability: {'STABLE' if stable else 'UNSTABLE'}\n")
    print("Baseline firing rates:")
    for name, r, rt in zip(POP_NAMES, r_star, h + W @ r_star):
        flag = " ← dead" if r < 0.1 else ""
        print(f"  {name:4s}: {r:6.2f} Hz{flag}")
    return r_star, stable
 

# Simulation

def run_simulation(
    T:           float = 500.0,   # total time (ms)
    dt:          float = 0.1,     # time step (ms)
    I_td:        float = 0.0,     # top-down modulatory input to VIP
    I_td_onset:  float = 200.0,   # time VIP modulation turns on
    I_td_offset: float = 350.0,   # time VIP modulation turns off
    noise_std:   float = 0.2,    # additive noise std
    r0:          np.ndarray = None  # initial firing rates [E, PV, SOM, VIP] # pyright: ignore[reportArgumentType]
):
    """
    Run Wilson-Cowan rate dynamics.
 
    Returns:
        t    — time vector [N]
        R    — firing rates [N, 4] — columns: E, PV, SOM, VIP
    """
    rng = np.random.default_rng(42) 

    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
 
    R = np.zeros((n_steps, 4))
    R[0] = np.array([5.0, 5.0, 5.0, 5.0]) if r0 is None else r0
 
    for i in range(n_steps - 1):
        r = R[i]
 
        # external inputs
        I_ext = np.zeros(4)                                
        I_ext[3] = I_td if I_td_onset <= t[i] <= I_td_offset else 0.0 # td input to VIP
 
        # total synaptic input to each population
        I_syn = W @ r + h + I_ext
 
        # noise
        I_syn += rng.normal(0.0, noise_std, 4)
 
        # Wilson-Cowan update: tau * dr/dt = -r + F(I_syn)
        dr       = (-r + relu(I_syn)) / TAU
        R[i + 1] = np.maximum(0.0, r + dt * dr)
 
    R = np.clip(R, 0, None)     # firing rates must be non-negative
    return t, R
 
 
# Analysis helpers
 
def run_vip_sweep(I_td_values, **sim_kwargs):
    """
    Sweep VIP modulatory input, record steady-state firing rates.
    Returns: I_td_values, steady_states [n_vals, 4]
    """
    steady = np.zeros((len(I_td_values), 4))
    for j, i_td in enumerate(I_td_values):
        _, R = run_simulation(T=300.0, I_td=i_td, I_td_onset=0.0,
                              I_td_offset=300.0, noise_std=0.0, **sim_kwargs)
        steady[j] = R[-1]
    return steady
 
 
# Plotting
 
COLORS = {
    "E": "#534AB7",    # purple
    "PV": "#993C1D",   # coral
    "SOM": "#0F6E56",  # teal
    "VIP": "#BA7517",  # amber
}
 
def plot_timecourse(t, h, R, I_td_onset=200.0, I_td_offset=350.0, title=""):
    r_fp = find_fp(W, h, R[0,:])  
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, name in enumerate(POP_NAMES):
        ax.plot(t, R[:, k], label=name, color=COLORS[name], lw=1.8)
        ax.axhline(r_fp[k], color=COLORS[name], lw=0.8, ls=":", alpha=0.4)

    ax.axvspan(I_td_onset, I_td_offset, alpha=0.08, color="gold",
               label="VIP input on")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title or "Disinhibitory circuit dynamics")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
 
 
def plot_gain_curve(I_td_values, steady):
    """Show E-cell rate as a function of VIP drive — the disinhibitory gain curve."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, name in enumerate(POP_NAMES):
        ax = axes[0] if name in ("E", "PV") else axes[1]
        ax.plot(I_td_values, steady[:, k], label=name, color=COLORS[name], lw=1.8)
    for ax, title in zip(axes, ["E and PV", "SOM and VIP"]):
        ax.set_xlabel("VIP modulatory input")
        ax.set_ylabel("Steady-state rate (Hz)")
        ax.set_title(title)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("Gain modulation via VIP disinhibitory pathway")
    plt.tight_layout()
    return fig
 
 
# Main
def main():
    # Pre-experiment analysis: fixed points and stability
    fp_report(W,h)

    # Experiment 1 (Sanity  check): compare no-VIP vs strong-VIP steady state
    print("\nSteady-state rates (no VIP input):")
    _, R_base = run_simulation(T=300, I_td=0.0,
                               I_td_onset=999, noise_std=0.0)
    for k, name in enumerate(POP_NAMES):
        print(f"  {name}: {R_base[-1, k]:.1f} Hz")
 
    print("\nSteady-state rates (VIP input = 3.0):")
    _, R_vip = run_simulation(T=300, I_td=3.0,
                              I_td_onset=0, I_td_offset=300, noise_std=0.0)
    for k, name in enumerate(POP_NAMES):
        print(f"  {name}: {R_vip[-1, k]:.1f} Hz")

    # Experiment 2: time course with VIP pulse
    print("Running experiment 2: transient VIP modulation...")
    t, R = run_simulation(
        T=500.0, dt=0.1,
        I_td=3.0, I_td_onset=200.0, I_td_offset=350.0,
        noise_std=0.05,
    )
    fig1 = plot_timecourse(t, h, R, title="VIP pulse → disinhibition of E population")
    fig1.savefig("figures/exp1_timecourse.png", dpi=150)
    print("  → Saved exp1_timecourse.png")
 
    # Experiment 3: gain curve (steady-state E rate vs VIP drive)
    print("Running experiment 3: VIP gain sweep...")
    I_td_values = np.linspace(0, 6, 60)
    steady = run_vip_sweep(I_td_values)
    fig2 = plot_gain_curve(I_td_values, steady)
    fig2.savefig("figures/exp2_gain_curve.png", dpi=150)
    print("  → Saved exp2_gain_curve.png")
 
    plt.show()
 
 
if __name__ == "__main__":
    main()