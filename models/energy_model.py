class EnergyModel:
    def __init__(self):
        self.E_slm = 1.0
        self.E_rl = 5.0
        self.tx_factor = 1.0

    def compute(self, power, source):
        # Transmission energy
        E_tx = self.tx_factor * power

        # Computation energy
        if source == "SLM":
            E_comp = self.E_slm
        else:
            E_comp = self.E_slm + self.E_rl

        return E_tx + E_comp