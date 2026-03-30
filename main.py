import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
 
 
# ── Connectivity matrix ───────────────────────────────────────────────────────
# W[i, j] = weight from population j → population i
# Sign convention: positive = excitatory, negative = inhibitory
# Populations indexed: 0=E, 1=PV, 2=SOM, 3=VIP
#
# Derived from Pfeffer et al. 2013 + Litwin-Kumar et al. 2016
# Key rules:
#   PV  → E, PV (strong self-inhibition)
#   SOM → E, PV, VIP (avoids SOM→SOM)
#   VIP → SOM (strong), PV (weak)
#   E   → PV, SOM, VIP (broad excitatory drive)
 
W = np.array([
    #  E      PV     SOM    VIP     ← presynaptic
    [ 0.5,  -1.5,  -1.0,   0.0 ],  # E   (post)
    [ 1.0,  -1.0,  -0.5,  -0.1 ],  # PV  (post)
    [ 0.8,   0.0,   0.0,  -1.5 ],  # SOM (post) — SOM→SOM absent
    [ 0.5,  -0.3,  -0.8,   0.0 ],  # VIP (post)
])
# Tweak these weights to explore gain modulation and disinhibitory gating.
# The key ratio to play with: W[SOM, VIP] / W[E, SOM]
 
 
# ── Neuron parameters ─────────────────────────────────────────────────────────
 
@dataclass
class PopulationParams:
    tau:        float   # time constant (ms)
    r_max:      float   # max firing rate (Hz)
    threshold:  float   # input threshold
    gain:       float   # sigmoid gain
 
 
PARAMS = {
    "E":   PopulationParams(tau=20.0, r_max=100.0, threshold=0.0, gain=1.2),
    "PV":  PopulationParams(tau=10.0, r_max=150.0, threshold=0.0, gain=1.5),
    "SOM": PopulationParams(tau=25.0, r_max=80.0,  threshold=0.0, gain=1.0),
    "VIP": PopulationParams(tau=20.0, r_max=80.0,  threshold=0.0, gain=1.0),
}
POP_NAMES = ["E", "PV", "SOM", "VIP"]
 
 
# ── Transfer function ─────────────────────────────────────────────────────────
 
def sigmoid(x, r_max, threshold, gain):
    """Rectified sigmoid transfer function (firing rate nonlinearity)."""
    return r_max / (1.0 + np.exp(-gain * (x - threshold)))
 
 
def transfer(x, pop: str):
    p = PARAMS[pop]
    return sigmoid(x, p.r_max, p.threshold, p.gain)
 
 
# ── Simulation ────────────────────────────────────────────────────────────────
 
def run_simulation(
    T:           float = 500.0,   # total time (ms)
    dt:          float = 0.1,     # time step (ms)
    I_ff:        float = 2.0,     # feedforward input to E (sensory drive)
    I_td:        float = 0.0,     # top-down modulatory input to VIP
    I_td_onset:  float = 200.0,   # time VIP modulation turns on
    I_td_offset: float = 350.0,   # time VIP modulation turns off
    noise_std:   float = 0.05,    # additive noise std
    r0:          np.ndarray = None,  # initial firing rates [E, PV, SOM, VIP]
):
    """
    Run Wilson-Cowan rate dynamics.
 
    Returns:
        t    — time vector [N]
        R    — firing rates [N, 4] — columns: E, PV, SOM, VIP
    """
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
 
    R = np.zeros((n_steps, 4))
    R[0] = r0 if r0 is not None else np.array([5.0, 5.0, 5.0, 2.0])
 
    for i in range(n_steps - 1):
        r = R[i]
 
        # external inputs
        I_ext = np.zeros(4)
        I_ext[0] = I_ff                                        # sensory → E
        I_ext[3] = I_td if I_td_onset <= t[i] <= I_td_offset else 0.0
 
        # total synaptic input to each population
        I_syn = W @ r + I_ext
 
        # noise
        I_syn += np.random.randn(4) * noise_std
 
        # Wilson-Cowan update: tau * dr/dt = -r + F(I_syn)
        for k, name in enumerate(POP_NAMES):
            tau = PARAMS[name].tau
            dr  = (-r[k] + transfer(I_syn[k], name)) / tau
            R[i + 1, k] = r[k] + dt * dr
 
    R = np.clip(R, 0, None)     # firing rates are non-negative
    return t, R
 
 
# ── Analysis helpers ──────────────────────────────────────────────────────────
 
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
 
 
# ── Plotting ──────────────────────────────────────────────────────────────────
 
COLORS = {
    "E": "#534AB7",    # purple
    "PV": "#993C1D",   # coral
    "SOM": "#0F6E56",  # teal
    "VIP": "#BA7517",  # amber
}
 
 
def plot_timecourse(t, R, I_td_onset=200.0, I_td_offset=350.0, title=""):
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, name in enumerate(POP_NAMES):
        ax.plot(t, R[:, k], label=name, color=COLORS[name], lw=1.8)
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
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    # ── Experiment 1: time course with VIP pulse
    print("Running experiment 1: transient VIP modulation...")
    t, R = run_simulation(
        T=500.0, dt=0.1,
        I_ff=2.0,
        I_td=3.0, I_td_onset=200.0, I_td_offset=350.0,
        noise_std=0.05,
    )
    fig1 = plot_timecourse(t, R, title="VIP pulse → disinhibition of E population")
    fig1.savefig("figures/exp1_timecourse.png", dpi=150)
    print("  → Saved exp1_timecourse.png")
 
    # ── Experiment 2: gain curve (steady-state E rate vs VIP drive)
    print("Running experiment 2: VIP gain sweep...")
    I_td_values = np.linspace(0, 6, 60)
    steady = run_vip_sweep(I_td_values, I_ff=2.0)
    fig2 = plot_gain_curve(I_td_values, steady)
    fig2.savefig("figures/exp2_gain_curve.png", dpi=150)
    print("  → Saved exp2_gain_curve.png")
 
    # ── Experiment 3: compare no-VIP vs strong-VIP steady state
    print("\nSteady-state rates (no VIP input):")
    _, R_base = run_simulation(T=300, I_ff=2.0, I_td=0.0,
                               I_td_onset=999, noise_std=0.0)
    for k, name in enumerate(POP_NAMES):
        print(f"  {name}: {R_base[-1, k]:.1f} Hz")
 
    print("\nSteady-state rates (VIP input = 3.0):")
    _, R_vip = run_simulation(T=300, I_ff=2.0, I_td=3.0,
                              I_td_onset=0, I_td_offset=300, noise_std=0.0)
    for k, name in enumerate(POP_NAMES):
        print(f"  {name}: {R_vip[-1, k]:.1f} Hz")
 
    plt.show()
 
 
if __name__ == "__main__":
    main()