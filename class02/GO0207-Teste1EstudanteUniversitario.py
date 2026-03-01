# GO0207-Teste1EstudanteUniversitário
# Perfil 1: Estudante universitário


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE 1: Estudante universitário")
    print("="*60)
    sistema.definir_contexto(
        clima='sol',
        orcamento='baixo',
        preferencia='outdoor',
        acompanhantes='amigos',
        dia='sábado'
    )
    sistema.inferir()
    sistema.mostrar_recomendacoes()

    # Perfil 2: Casal com criança pequena
    print("\n" + "="*60)
    print("TESTE 2: Família com criança")
    print("="*60)
    sistema.definir_contexto(
        clima='chuva',
        orcamento='medio',
        preferencia='indoor',
        acompanhantes='família',
        dia='domingo'
    )
    sistema.inferir()
    sistema.mostrar_recomendacoes()

    # Perfil 3: Profissional em busca de relaxamento
    print("\n" + "="*60)
    print("TESTE 3: Profissional estressado")
    print("="*60)
    sistema.definir_contexto(
        clima='nublado',
        orcamento='alto',
        preferencia='relax',
        acompanhantes='sozinho',
        dia='sábado'
    )
    sistema.inferir()
    sistema.mostrar_recomendacoes()
