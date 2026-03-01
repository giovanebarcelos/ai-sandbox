# GO1101-Png
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


if __name__ == '__main__':
    print("=== Função de Pertinência Triangular ===")
    # Parâmetros: pé esquerdo a=0, pico b=5, pé direito c=10
    a, b, c = 0, 5, 10
    print(f"  Parâmetros: a={a}, b={b}, c={c}")
    print()
    for x in range(-1, 12):
        mu = triangular(x, a, b, c)
        barra = "█" * int(mu * 20)
        print(f"  x={x:3d}: μ={mu:.2f} |{barra}")
