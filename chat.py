import os
import torch
import tiktoken
from rag import (
    answer_question_with_rag,
    RAG_TOP_K
)
from gpt_download import download_and_load_gpt2

from previous_chapters import (
    GPTModel,
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text
)



# SETTINGS

MODEL_PATH = "qa_finetuned_gpt2_medium.pth"

CHOOSE_MODEL = "gpt2-medium (355M)"


# DEVICE

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")


print(f"Using device: {device}")


# TOKENIZER

tokenizer = tiktoken.get_encoding("gpt2")


# MODEL CONFIGURATION

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}


model_configs = {
    "gpt2-small (124M)": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    },

    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    },

    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20
    },

    "gpt2-xl (1558M)": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25
    }
}


BASE_CONFIG.update(
    model_configs[CHOOSE_MODEL]
)


# BUILD MODEL

model = GPTModel(BASE_CONFIG)

# LOAD FINE-TUNED CHECKPOINT

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Fine-tuned model checkpoint not found: {MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)


model.to(device)
model.eval()

print(f"Loaded fine-tuned model: {MODEL_PATH}")


# QA PROMPT

def format_input(question):
    return (
        "You are an NLP course assistant. "
        "Answer the following question accurately and clearly."
        f"\n\n### Question:\n{question.strip()}"
    )


# INTERACTIVE CHAT

print("\n========== Prototype Chat ==========")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Chat ended.")
        break

    if not question:
        continue

    rag_result = answer_question_with_rag(
        question=question,
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_size=BASE_CONFIG["context_length"],
        top_k=RAG_TOP_K,
        max_new_tokens=96,
        answer_mode="synthesized"
    )

    print(
        f"\nModel: "
        f"{rag_result['answer']}\n"
    )

    print("Retrieved sources:")

    for chunk in rag_result["retrieved_chunks"]:
        print(
            f"- {chunk['source']} "
            f"[Section: {chunk.get('section', 'General')}] "
            f"(similarity: {chunk['score']:.4f})"
        )

    print()
