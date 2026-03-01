# GO1643-Codigo
# GPT-2/GPT-J:


if __name__ == "__main__":
    target_modules = ["c_attn", "c_proj"]

    # Llama/Mistral:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # BERT:
    target_modules = ["query", "key", "value"]

    # Todos (max performance, mais parâmetros):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
