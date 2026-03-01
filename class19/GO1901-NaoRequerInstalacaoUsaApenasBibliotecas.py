# GO1901-NãoRequerInstalaçãoUsaApenasBibliotecas
# Cada gene é 0 ou 1
cromossomo = [1, 0, 1, 1, 0, 1, 0, 0]
# Exemplo: mochila (1=item incluído, 0=excluído)

# Decodificar para valor real
def decode_binary(chromosome, min_val, max_val):
    # Converter binário para decimal
    decimal = int(''.join(map(str, chromosome)), 2)
    # Mapear para intervalo [min_val, max_val]
    max_decimal = 2**len(chromosome) - 1
    value = min_val + (max_val - min_val) * decimal / max_decimal
    return value

# [1,0,1,1,0] = 22 decimal → mapear [0,10] = 8.57
