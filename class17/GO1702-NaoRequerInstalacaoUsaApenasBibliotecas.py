# GO1702-NãoRequerInstalaçãoUsaApenasBibliotecas


if __name__ == "__main__":
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[
            {"source": "manual.pdf", "page": 1, "date": "2025-01"},
            {"source": "manual.pdf", "page": 2, "date": "2025-01"}
        ]
    )
