# GO1240-GradienteDurante
# Gradiente durante backprop:
∂Loss/∂x = ∂Loss/∂H × ∂H/∂x
         = ∂Loss/∂H × (∂F/∂x + 1)  # "+1" vem da skip connection

# Skip connection = "autoestrada" para gradiente!
# Não passa por ativações (não sofre saturação)
