# GO1611-21ProjetoChatbotComGpt2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(prompt, max_length=100, temperature=0.7):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

    prompt = "Once upon a time in a magical forest,"
    story = generate_text(prompt, max_length=200, temperature=0.8)
    print(story)

    def chatbot():
        print("Chatbot GPT-2 (digite 'sair' para encerrar)")
        conversation = ""
        while True:
            user_input = input("Você: ")
            if user_input.lower() == 'sair':
                break

            conversation += f"User: {user_input}\nBot: "
            response = generate_text(conversation, max_length=150)
            bot_response = response[len(conversation):].split('\n')[0]
            conversation += bot_response + "\n"

            print(f"Bot: {bot_response}")
