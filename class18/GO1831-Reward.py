# GO1831-Reward
reward = + SpO2_improvement  # Aumentar saturação oxigênio
         - lung_damage_risk  # Evitar barotrauma
         - ventilator_days   # Minimizar tempo em ventilação
         + 100 if weaning_success  # Sucesso em retirar ventilação
