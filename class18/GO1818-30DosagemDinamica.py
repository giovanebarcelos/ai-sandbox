# GO1818-30DosagemDinâmica
class ChemotherapyEnv:
    def __init__(self, patient_profile):
        self.cancer_cells = 1e9  # Células iniciais
        self.toxicity = 0
        self.patient_health = 100

    def step(self, dose_percentage):
        # dose: 0.0 (0%) a 1.0 (100%)

        # Efeito no câncer (dose maior mata mais)
        kill_rate = dose_percentage * 0.3  # 30% por dose completa
        self.cancer_cells *= (1 - kill_rate)

        # Efeito colateral (toxicidade acumula)
        self.toxicity += dose_percentage * 20
        self.patient_health -= dose_percentage * 15

        # Recompensa complexa
        reward = 0
        reward += 10 * kill_rate  # Matar células cancerígenas é bom
        reward -= self.toxicity * 0.5  # Toxicidade é ruim
        reward -= 100 if self.patient_health < 20 else 0  # Não matar paciente!

        # Termina se curou ou morreu
        done = (self.cancer_cells < 1e6) or (self.patient_health < 0)

        state = [self.cancer_cells, self.toxicity, self.patient_health]
        return state, reward, done

# Treinar com DQN ou PPO
# Política aprendida: Dose alta no início, reduzir gradualmente
