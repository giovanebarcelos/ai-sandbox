# GO1703-NãoRequerInstalaçãoUsaApenasBibliotecas


if __name__ == "__main__":
    results = vectorstore.similarity_search(
        "política de férias",
        filter={"source": "manual.pdf", "date": "2025-01"},
        k=3
    )
