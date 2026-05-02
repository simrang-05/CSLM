import numpy as np

SINR_THRESHOLD = 1.0  # Γ


import numpy as np

SINR_THRESHOLD = 1.0  # Γ

def joint_optimize(state, power, channel):

    h = state[channel]
    interference = state[8]
    noise = state[9]

    # Avoid division issues
    h = max(h, 1e-6)

    # Required power to satisfy SINR
    required_power = SINR_THRESHOLD * (interference + noise) / h

    # Final power (respect original decision but enforce constraint)
    P = max(power, required_power)

    # Clip to system limits
    P = np.clip(P, 0.1, 1.0)

    return float(P), int(channel)