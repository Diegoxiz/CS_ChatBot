from importlib.metadata import version
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import requests

KNOWLEDGE_BASE_DIRECTORY = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "MODEL_KNOWLEDGE_BASE"
)

# 1 DATASET DOWNLOAD AND FORMATTING ========================================

pkgs = [
    "numpy",                  # PyTorch and TensorFlow dependency
    "matplotlib",             # Plotting library
    "tiktoken",               # Tokenizer
    "torch",                  # Deep learning library
    "tqdm",                   # Progress bar
    "tensorflow",             # For OpenAI pretrained weights
    "sentence-transformers"   # Embedding model for RAG
]


# MODIFIED:
# The prototype dataset is stored as JSONL instead of a JSON array.
DATASET_DIRECTORY = r"C:\Users\diego\Desktop\MODEL_KNOWLEDGE_BASE"


def load_jsonl(file_path):
    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                entry = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {file_path}: {error}"
                ) from error

            required_fields = {
                "instruction",
                "input",
                "output"
            }

            missing_fields = required_fields - entry.keys()

            if missing_fields:
                raise ValueError(
                    f"Missing fields {missing_fields} "
                    f"on line {line_number} of {file_path}"
                )

            data.append(entry)

    return data


def load_all_qa_jsonl(root_directory):
    data = []

    root_path = Path(root_directory)

    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root_directory}"
        )

    jsonl_files = sorted(
        root_path.glob("*/qa/*.jsonl")
    )

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files were found inside subject qa folders: "
            f"{root_directory}"
        )

    for file_path in jsonl_files:
        subject_name = file_path.parent.parent.name

        file_records = load_jsonl(file_path)

        for record in file_records:
            record["subject"] = subject_name
            record["source_file"] = file_path.name

        data.extend(file_records)

        print(
            f"Loaded {len(file_records)} records from "
            f"{subject_name}/{file_path.name}"
        )

    print(f"\nTotal QA records loaded: {len(data)}")

    return data


data = load_all_qa_jsonl(DATASET_DIRECTORY)


# MODIFIED:
# The prompt is now explicitly question-answer oriented.
def format_input(entry):
    question_text = entry["instruction"].strip()
    additional_input = entry.get("input", "").strip()

    prompt_text = (
        "You are an NLP course assistant. "
        "Answer the following question accurately "
        "and clearly."
        f"\n\n### Question:\n"
        f"{question_text}"
    )

    if additional_input:
        prompt_text += (
            f"\n\n### Additional Input:\n"
            f"{additional_input}"
        )

    prompt_text += "\n\n### Answer:\n"

    return prompt_text


train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = (
    len(data)
    - train_portion
    - test_portion
)  # Remaining 5% for validation

train_data = data[:train_portion]

test_data = data[
    train_portion:
    train_portion + test_portion
]

val_data = data[
    train_portion + test_portion:
]


# 2 BATCHING THE DATASET ==================================================

import torch
from torch.utils.data import Dataset


class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []

        for entry in data:
            question_plus_context = format_input(entry)

            # MODIFIED:
            # The output label is now "Answer".
            answer_text = (
                f"\n\n### Answer:\n{entry['output']}"
            )

            full_text = (
                question_plus_context
                + answer_text
            )

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(
        len(item) + 1
        for item in batch
    )

    # Pad and prepare inputs and targets
    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        # Add an <|endoftext|> token
        new_item += [pad_token_id]

        # Pad sequences to max_length
        padded = (
            new_item
            + [pad_token_id]
            * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(
            padded[:-1]
        )

        targets = torch.tensor(
            padded[1:]
        )

        # Replace all but the first padding tokens
        # in targets with ignore_index
        mask = targets == pad_token_id

        indices = torch.nonzero(
            mask
        ).squeeze()

        if indices.numel() > 1:
            targets[
                indices[1:]
            ] = ignore_index

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[
                :allowed_max_length
            ]

            targets = targets[
                :allowed_max_length
            ]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert lists to tensors and transfer to device
    inputs_tensor = torch.stack(
        inputs_lst
    ).to(device)

    targets_tensor = torch.stack(
        targets_lst
    ).to(device)

    return inputs_tensor, targets_tensor


# 3 CREATING DATA LOADERS ================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    # Use PyTorch 2.9 or newer for stable MPS results
    major, minor = map(
        int,
        torch.__version__.split(".")[:2]
    )

    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

else:

    device = torch.device("cpu")


from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)


from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)


train_dataset = InstructionDataset(
    train_data,
    tokenizer
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)


val_dataset = InstructionDataset(
    val_data,
    tokenizer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


test_dataset = InstructionDataset(
    test_data,
    tokenizer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


# 4 LOADING A PRETRAINED LLM =============================================

from gpt_download import download_and_load_gpt2

from previous_chapters import (
    GPTModel,
    load_weights_into_gpt
)

# If the previous_chapters.py file is not available locally,
# you can import it from the llms-from-scratch PyPI package.
#
# from llms_from_scratch.ch04 import GPTModel
# from llms_from_scratch.ch05 import (
#     download_and_load_gpt2,
#     load_weights_into_gpt
# )


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


CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(
    model_configs[CHOOSE_MODEL]
)


model_size = (
    CHOOSE_MODEL
    .split(" ")[-1]
    .lstrip("(")
    .rstrip(")")
)


settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)


model = GPTModel(BASE_CONFIG)

load_weights_into_gpt(
    model,
    params
)

model.eval()


torch.manual_seed(123)

input_text = format_input(
    val_data[0]
)


from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# Alternatively:
#
# from llms_from_scratch.ch05 import (
#     generate,
#     text_to_token_ids,
#     token_ids_to_text
# )


token_ids = generate(
    model=model,
    idx=text_to_token_ids(
        input_text,
        tokenizer
    ),
    max_new_tokens=35,
    context_size=BASE_CONFIG[
        "context_length"
    ],
    eos_id=50256
)


generated_text = token_ids_to_text(
    token_ids,
    tokenizer
)


# MODIFIED:
# Remove the QA answer heading instead of the response heading.
response_text = (
    generated_text[
        len(input_text):
    ]
    .replace(
        "### Answer:",
        ""
    )
    .strip()
)

print(response_text)


# 5 QUESTION-ANSWER FINETUNING THE LLM ===================================

# THIS PART HAS BEEN MODIFIED TO AVOID
# THE TRAINING PART EVERY TIME IT RUNS

from previous_chapters import (
    calc_loss_loader,
    train_model_simple
)


TRAIN_MODEL = False

MODEL_PATH = (
    "qa_finetuned_gpt2_medium.pth"
)


if TRAIN_MODEL:

    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():

        train_loss = calc_loss_loader(
            train_loader,
            model,
            device,
            num_batches=5
        )

        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            num_batches=5
        )

    print(
        "Training loss:",
        train_loss
    )

    print(
        "Validation loss:",
        val_loss
    )

    import time

    start_time = time.time()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00005,
        weight_decay=0.1
    )

    num_epochs = 2

    (
        train_losses,
        val_losses,
        tokens_seen
    ) = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(
            val_data[0]
        ),
        tokenizer=tokenizer
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    end_time = time.time()

    execution_time_minutes = (
        end_time - start_time
    ) / 60

    print(
        f"Training completed in "
        f"{execution_time_minutes:.2f} minutes."
    )

    print("Model saved.")


else:

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    print("Loaded fine-tuned model.")


# MODIFIED:
# Plot training losses only when training occurred.
if TRAIN_MODEL:

    from previous_chapters import plot_losses

    # Alternatively:
    #
    # from llms_from_scratch.ch05 import plot_losses

    epochs_tensor = torch.linspace(
        0,
        num_epochs,
        len(train_losses)
    )

    plot_losses(
        epochs_tensor,
        tokens_seen,
        train_losses,
        val_losses
    )


# 8 EXTRACTING RESPONSES =================================================

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(
            input_text,
            tokenizer
        ).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG[
            "context_length"
        ],
        eos_id=50256
    )

    generated_text = token_ids_to_text(
        token_ids,
        tokenizer
    )

    response_text = (
        generated_text[
            len(input_text):
        ]
        .replace(
            "### Answer:",
            ""
        )
        .strip()
    )

    print(input_text)

    print(
        f"\nCorrect answer:\n"
        f">> {entry['output']}"
    )

    print(
        f"\nModel answer:\n"
        f">> {response_text}"
    )

    print(
        "-------------------------------------"
    )


from tqdm import tqdm


for i, entry in tqdm(
    enumerate(test_data),
    total=len(test_data)
):

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(
            input_text,
            tokenizer
        ).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG[
            "context_length"
        ],
        eos_id=50256
    )

    generated_text = token_ids_to_text(
        token_ids,
        tokenizer
    )

    response_text = (
        generated_text[
            len(input_text):
        ]
        .replace(
            "### Answer:",
            ""
        )
        .strip()
    )

    test_data[i][
        "model_response"
    ] = response_text


with open(
    "qa-data-with-response.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        test_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# 9 RETRIEVAL-AUGMENTED GENERATION =======================================

import numpy as np

from pathlib import Path

from sentence_transformers import (
    SentenceTransformer
)


# Change this path only if the cleaned notes
# are stored in a different directory.
CLEANED_NOTES_DIRECTORY = DATASET_DIRECTORY


RAG_EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


RAG_CHUNK_SIZE = 250
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3


def load_cleaned_documents(directory):
    documents = []

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"The cleaned-notes directory "
            f"does not exist: {directory}"
        )

    text_files = []

    for subject_folder in directory_path.iterdir():

        if not subject_folder.is_dir():
            continue

        cleaned_folder = subject_folder / "cleaned"

        if cleaned_folder.exists():
            text_files.extend(
                sorted(cleaned_folder.glob("*.txt"))
            )

    if not text_files:
        raise FileNotFoundError(
            f"No .txt files were found "
            f"inside: {directory}"
        )

    for file_path in text_files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

        if text:

            documents.append({
                "source": str(file_path),
                "text": text
            })

    return documents


def create_document_chunks(
    documents,
    tokenizer,
    chunk_size=250,
    chunk_overlap=50
):
    chunks = []

    step_size = (
        chunk_size
        - chunk_overlap
    )

    if step_size <= 0:
        raise ValueError(
            "chunk_overlap must be "
            "smaller than chunk_size."
        )

    for document in documents:

        token_ids = tokenizer.encode(
            document["text"]
        )

        for start_index in range(
            0,
            len(token_ids),
            step_size
        ):

            end_index = (
                start_index
                + chunk_size
            )

            chunk_token_ids = token_ids[
                start_index:end_index
            ]

            if not chunk_token_ids:
                continue

            chunk_text = tokenizer.decode(
                chunk_token_ids
            ).strip()

            if not chunk_text:
                continue

            chunks.append({
                "source": document[
                    "source"
                ],
                "text": chunk_text
            })

            if end_index >= len(token_ids):
                break

    return chunks


cleaned_documents = load_cleaned_documents(
    CLEANED_NOTES_DIRECTORY
)


document_chunks = create_document_chunks(
    cleaned_documents,
    tokenizer,
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP
)


embedding_model = SentenceTransformer(
    RAG_EMBEDDING_MODEL
)


chunk_texts = [
    chunk["text"]
    for chunk in document_chunks
]


chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


def retrieve_relevant_chunks(
    question,
    top_k=RAG_TOP_K
):
    question_embedding = (
        embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
    )

    similarity_scores = (
        chunk_embeddings
        @ question_embedding
    )

    top_indices = np.argsort(
        similarity_scores
    )[-top_k:][::-1]

    retrieved_chunks = []

    for index in top_indices:

        retrieved_chunks.append({
            "source": document_chunks[
                index
            ]["source"],

            "text": document_chunks[
                index
            ]["text"],

            "score": float(
                similarity_scores[index]
            )
        })

    return retrieved_chunks


print("\n========== Prototype Chat ==========\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    prompt = format_input({
        "instruction": question,
        "input": ""
    })

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(
            prompt,
            tokenizer
        ).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )

    generated = token_ids_to_text(
        token_ids,
        tokenizer
    )

    answer = (
        generated[len(prompt):]
        .replace("### Answer:", "")
        .strip()
    )

    print("\nModel:", answer)
    print()

#RAG SECTION
def format_rag_input(
    question,
    retrieved_chunks
):
    context_sections = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_sections.append(
            f"[Course Note {index}]\n"
            f"{chunk['text']}"
        )

    retrieved_context = (
        "\n\n".join(
            context_sections
        )
    )

    prompt_text = (
        "You are an NLP course assistant. "
        "Answer the following question accurately "
        "and clearly using the supplied course notes. "
        "Do not invent information that is not "
        "supported by the notes."
        f"\n\n### Context:\n"
        f"{retrieved_context}"
        f"\n\n### Question:\n"
        f"{question.strip()}"
    )

    return prompt_text


def trim_rag_prompt(
    prompt_text,
    tokenizer,
    max_prompt_tokens=768
):
    prompt_token_ids = tokenizer.encode(
        prompt_text
    )

    if (
        len(prompt_token_ids)
        <= max_prompt_tokens
    ):
        return prompt_text

    prompt_token_ids = prompt_token_ids[
        -max_prompt_tokens:
    ]

    return tokenizer.decode(
        prompt_token_ids
    )


def answer_question_with_rag(
    question,
    model,
    tokenizer,
    device,
    top_k=RAG_TOP_K,
    max_new_tokens=256
):
    retrieved_chunks = (
        retrieve_relevant_chunks(
            question,
            top_k=top_k
        )
    )

    input_text = format_rag_input(
        question,
        retrieved_chunks
    )

    input_text = trim_rag_prompt(
        input_text,
        tokenizer,
        max_prompt_tokens=(
            BASE_CONFIG[
                "context_length"
            ]
            - max_new_tokens
        )
    )

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(
            input_text,
            tokenizer
        ).to(device),
        max_new_tokens=max_new_tokens,
        context_size=BASE_CONFIG[
            "context_length"
        ],
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.15,
        eos_id=50256
    )

    generated_text = token_ids_to_text(
        token_ids,
        tokenizer
    )

    answer_text = (
        generated_text[
            len(input_text):
        ]
        .replace(
            "### Answer:",
            ""
        )
        .strip()
    )

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": (
            retrieved_chunks
        )
    }

if __name__ == "__main__":
    # Example RAG question
    rag_question = (
        "What is Word2Vec and how does it "
        "differ from Bag-of-Words?"
    )


    rag_result = answer_question_with_rag(
        question=rag_question,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=RAG_TOP_K,
        max_new_tokens=256
    )


    print(
        "\nRAG QUESTION:\n"
        f"{rag_result['question']}"
    )

    print(
        "\nRAG ANSWER:\n"
        f"{rag_result['answer']}"
    )

    print(
        "\nRETRIEVED SOURCES:"
    )


    for retrieved_chunk in (
        rag_result["retrieved_chunks"]
    ):

        print(
            f"- {retrieved_chunk['source']} "
            f"(similarity: "
            f"{retrieved_chunk['score']:.4f})"
        )
