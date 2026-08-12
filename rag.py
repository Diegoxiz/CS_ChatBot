from pathlib import Path
import os
import re

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)


# CONFIGURATION

RAG_TOP_K = 2
RAG_CHUNK_SIZE = 180
RAG_CHUNK_OVERLAP = 30
RAG_MIN_SCORE = 0.25
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# For portability, the path can be overridden with the MODEL_KNOWLEDGE_BASE
# environment variable. The fallback keeps compatibility with the current PC.
DATASET_DIRECTORY = Path(
    os.environ.get(
        "MODEL_KNOWLEDGE_BASE",
        r"C:\Users\diego\Desktop\MODEL_KNOWLEDGE_BASE",
    )
)

CLEANED_FOLDER_NAME = "cleaned"

tokenizer = tiktoken.get_encoding("gpt2")


COMPARISON_ATTRIBUTES = {
    "output": [
        "predicts",
        "prediction",
        "output",
        "continuous",
        "probability",
        "probabilities",
        "classes",
        "real value",
    ],
    "task": [
        "regression",
        "classification",
        "task",
        "algorithm",
        "suitable for",
    ],
    "loss": [
        "loss",
        "least squares",
        "cross entropy",
        "binary cross entropy",
        "mse",
    ],
    "speed": [
        "faster",
        "slower",
        "speed",
        "efficient",
        "computational",
    ],
    "data": [
        "small dataset",
        "large dataset",
        "rare words",
        "frequent words",
        "data",
    ],
    "prediction_direction": [
        "center word",
        "target word",
        "surrounding words",
        "context words",
        "predicts context",
        "predicts the target",
    ],
}

INVALID_SECTION_NAMES = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "a",
    "an",
    "for",
    "with",
    "on",
    "by",
    "is",
    "are",
}

IMPLEMENTATION_SECTION_TERMS = {
    "gensim",
    "notebook",
    "code",
    "implementation",
    "switch",
    "example",
}

SECTION_KEYWORDS = {
    "limitations": [
        "limitation",
        "limitations",
        "disadvantage",
        "disadvantages",
        "drawback",
        "drawbacks",
    ],
    "advantages": [
        "advantage",
        "advantages",
        "benefit",
        "benefits",
    ],
    "comparison": [
        "difference",
        "differences",
        "compare",
        "comparison",
        "versus",
        " vs ",
    ],
    "definition": [
        "what is",
        "define",
        "definition",
    ],
}

# Aliases improve retrieval when a question uses an abbreviation while the notes
# use the full term or vice versa.
TERM_ALIASES = {
    "cbow": ["cbow", "continuous bag of words"],
    "continuous bag of words": ["continuous bag of words", "cbow"],
    "skip gram": ["skip gram", "skip-gram"],
    "skip-gram": ["skip gram", "skip-gram"],
    "bag of words": ["bag of words", "bag-of-words", "bow"],
    "bag-of-words": ["bag of words", "bag-of-words", "bow"],
    "bow": ["bag of words", "bag-of-words", "bow"],
    "word2vec": ["word2vec", "word 2 vec"],
    "rnn": ["rnn", "rnns", "recurrent neural network", "recurrent neural networks"],
    "rnns": ["rnn", "rnns", "recurrent neural network", "recurrent neural networks"],
    "lstm": ["lstm", "lstms", "long short term memory"],
    "lstms": ["lstm", "lstms", "long short term memory"],
    "hmm": ["hmm", "hidden markov model", "hidden markov models"],
    "bert": ["bert"],
    "gpt": ["gpt"],
}


# TEXT HELPERS!!!!!

def normalize_for_match(text):
    # Normalizes punctuation and whitespace so terms can be compared.
    text = text.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_term_aliases(term):
    # Returns known aliases for a comparison term, including the original term.
    normalized = normalize_for_match(term)
    aliases = TERM_ALIASES.get(normalized, [term])

    result = []
    for alias in aliases + [term]:
        normalized_alias = normalize_for_match(alias)
        if normalized_alias and normalized_alias not in result:
            result.append(normalized_alias)

    return result


def display_term(term):
    # Formats a term for user-facing synthesized answers while preserving acronyms.
    normalized = term.strip()
    acronym_map = {
        "cbow": "CBOW",
        "rnn": "RNN",
        "rnns": "RNNs",
        "lstm": "LSTM",
        "lstms": "LSTMs",
        "hmm": "HMM",
        "bert": "BERT",
        "gpt": "GPT",
        "word2vec": "Word2Vec",
        "bag-of-words": "Bag-of-Words",
        "bag of words": "Bag-of-Words",
        "skip-gram": "Skip-Gram",
        "skip gram": "Skip-Gram",
    }

    return acronym_map.get(normalized.lower(), normalized.title())


def detect_question_intent(question):
    # Detects whether a question asks for a comparison, limitation, advantage, or definition.
    normalized_question = question.lower()

    for intent, keywords in SECTION_KEYWORDS.items():
        if any(keyword in normalized_question for keyword in keywords):
            return intent

    return "general"


def extract_comparison_terms(question):
    # Extracts the two concepts being compared from common comparison phrasings.
    normalized = question.strip()

    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)",
        r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)",
        r"compare\s+(.+?)\s+to\s+(.+?)(?:\?|\.|$)",
        r"compare\s+(.+?)\s+with\s+(.+?)(?:\?|\.|$)",
        r"(.+?)\s+versus\s+(.+?)(?:\?|\.|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|\.|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return [match.group(1).strip(), match.group(2).strip()]

    return []


# DOCUMENT LOADING / SECTIONING

def load_cleaned_documents(dataset_directory):
    # Loads every cleaned .txt note found inside subject-level cleaned folders.
    dataset_directory = Path(dataset_directory)

    if not dataset_directory.exists():
        raise FileNotFoundError(
            "The dataset directory does not exist: "
            f"{dataset_directory}\n"
            "Set the MODEL_KNOWLEDGE_BASE environment variable or update "
            "DATASET_DIRECTORY in rag.py."
        )

    documents = []

    for cleaned_directory in dataset_directory.rglob(CLEANED_FOLDER_NAME):
        if not cleaned_directory.is_dir():
            continue

        for file_path in cleaned_directory.rglob("*.txt"):
            text = file_path.read_text(encoding="utf-8").strip()

            if not text:
                continue

            documents.append({
                "source": str(file_path),
                "text": text,
            })

    if not documents:
        raise RuntimeError(
            "No cleaned text documents were found under: "
            f"{dataset_directory}"
        )

    return documents


def normalize_heading(line):
    # Removes Markdown/separator formatting from a possible section heading.
    line = re.sub(r"^#{1,6}\s*", "", line.strip())
    line = re.sub(r"[-=_*#~]+", " ", line)
    return re.sub(r"\s+", " ", line).strip(" :")


def is_valid_heading(heading):
    # Rejects tiny, stop-word-only, or sentence-like strings as section headings.
    heading = heading.strip()

    if not heading:
        return False

    normalized = heading.lower()

    if normalized in INVALID_SECTION_NAMES:
        return False

    if len(heading) < 3:
        return False

    if len(heading.split()) > 10:
        return False

    if re.fullmatch(r"\d+[.)]?", heading):
        return False

    # A trailing sentence-ending period is usually prose, not a heading.
    if heading.endswith(".") and len(heading.split()) > 3:
        return False

    return True


def is_probable_heading(line):
    line = line.strip()

    if not line:
        return False

    if re.fullmatch(r"[-=_*#~]{3,}", line):
        return False

    if re.match(r"^#{1,6}\s+\S", line):
        return True

    words = line.rstrip(":").split()

    if not 1 <= len(words) <= 10:
        return False

    if line.endswith(":"):
        return True

    if line.isupper():
        return True

    small_words = {
        "of", "and", "the", "vs", "with", "in", "for", "to", "or", "from"
    }

    return all(
        word[:1].isupper() or word.lower() in small_words
        for word in words
    )

#This part was a headache :(
def split_document_into_sections(document):
    # Splits a cleaned note into named sections while preserving line breaks and tables.
    sections = []
    current_heading = "General"
    current_lines = []

    for raw_line in document["text"].splitlines():
        line = raw_line.strip()

        if not line:
            # Preserve a single blank line inside a section because it can help table/prose structure.
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if re.fullmatch(r"[-=_*#~]{3,}", line):
            continue

        if is_probable_heading(line):
            normalized_heading = normalize_heading(line)

            if not is_valid_heading(normalized_heading):
                current_lines.append(line)
                continue

            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append({
                        "source": document["source"],
                        "section": current_heading,
                        "text": section_text,
                    })

            current_heading = normalized_heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append({
                "source": document["source"],
                "section": current_heading,
                "text": section_text,
            })

    return sections


def clean_retrieved_text(text, preserve_newlines=False):
    # Removes decorative artifacts while optionally preserving table/paragraph line breaks.
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            if preserve_newlines and cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if re.fullmatch(
            r"\[\[.*?(page|index).*?\]\]",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if re.fullmatch(r"[-=_*#~]{3,}", line):
            continue

        # Keep Markdown table separator rows because the table parser needs them.
        if "|" not in line:
            line = re.sub(r"^[\-=_*#~\s]+", "", line)
            line = re.sub(r"[\-=_*#~\s]+$", "", line)

        if not line:
            continue

        normalized = line.lower().rstrip(":")
        ignored_labels = {
            "course",
            "definition",
            "overview",
            "introduction",
            "summary",
            "contents",
            "table of contents",
            "page index",
        }

        if normalized in ignored_labels:
            continue

        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    if preserve_newlines:
        # Collapse repeated blank lines but keep meaningful row/paragraph boundaries.
        result_lines = []
        for line in cleaned_lines:
            if line == "" and result_lines and result_lines[-1] == "":
                continue
            result_lines.append(line)
        return "\n".join(result_lines).strip()

    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def create_document_chunks(
    documents,
    tokenizer,
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
):
    # Converts section-aware notes into overlapping token chunks for semantic retrieval.
    chunks = []
    step_size = chunk_size - chunk_overlap

    if step_size <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    for document in documents:
        sections = split_document_into_sections(document)
        topic_name = Path(document["source"]).stem.replace("_", " ")

        for section in sections:
            cleaned_section_text = clean_retrieved_text(
                section["text"],
                preserve_newlines=True,
            )

            if not cleaned_section_text:
                continue

            token_ids = tokenizer.encode(cleaned_section_text)
            prefix = (
                f"Topic: {topic_name}. "
                f"Section: {section['section']}. "
            )

            for start_index in range(0, len(token_ids), step_size):
                chunk_token_ids = token_ids[
                    start_index:start_index + chunk_size
                ]

                if not chunk_token_ids:
                    continue

                chunk_text = tokenizer.decode(chunk_token_ids).strip()

                if not chunk_text:
                    continue

                chunks.append({
                    "source": section["source"],
                    "section": section["section"],
                    "topic": topic_name,
                    "text": chunk_text,
                    "embedding_text": prefix + chunk_text,
                })

                if start_index + chunk_size >= len(token_ids):
                    break

    return chunks


# BUILD THE RETRIEVAL INDEX

cleaned_documents = load_cleaned_documents(DATASET_DIRECTORY)

document_chunks = create_document_chunks(
    cleaned_documents,
    tokenizer,
)

embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)

chunk_texts = [
    chunk["embedding_text"]
    for chunk in document_chunks
]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# RETRIEVAL

def chunk_contains_term(chunk, term):
    # Checks whether a chunk contains a comparison term or one of its aliases.
    section_text = normalize_for_match(chunk.get("section", ""))
    source_text = normalize_for_match(Path(chunk["source"]).stem)
    chunk_text = normalize_for_match(chunk["text"])

    aliases = get_term_aliases(term)

    return any(
        alias in section_text
        or alias in source_text
        or alias in chunk_text
        for alias in aliases
    )


def comparison_match_score(chunk, term, semantic_score):
    # Reranks a chunk for one comparison term, favoring conceptual sections over code notes.
    section_text = normalize_for_match(chunk.get("section", ""))
    source_text = normalize_for_match(Path(chunk["source"]).stem)
    chunk_text = normalize_for_match(chunk["text"])
    aliases = get_term_aliases(term)

    score = float(semantic_score)

    if any(alias in section_text for alias in aliases):
        score += 0.30

    if any(alias in source_text for alias in aliases):
        score += 0.15

    if any(alias in chunk_text for alias in aliases):
        score += 0.05

    section_words = set(section_text.split())
    implementation_hits = sum(
        normalize_for_match(term_name) in section_text
        for term_name in IMPLEMENTATION_SECTION_TERMS
    )
    score -= 0.20 * implementation_hits

    # Prefer short, concept-like section titles over long instructional headings.
    if 1 <= len(section_words) <= 5:
        score += 0.03

    return score


def find_best_chunk_for_term(term, similarity_scores):
    # Finds the strongest conceptual chunk for one side of a comparison question.
    candidates = []

    for index, chunk in enumerate(document_chunks):
        if not chunk_contains_term(chunk, term):
            continue

        adjusted_score = comparison_match_score(
            chunk,
            term,
            similarity_scores[index],
        )
        candidates.append((adjusted_score, index))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def find_complete_comparison_chunk(comparison_terms, similarity_scores):
    # Finds a single section that explicitly contains both sides of a comparison.
    if len(comparison_terms) < 2:
        return None

    first_term, second_term = comparison_terms[:2]
    candidates = []

    for index, chunk in enumerate(document_chunks):
        section_text = normalize_for_match(chunk.get("section", ""))
        text = normalize_for_match(chunk["text"])

        first_aliases = get_term_aliases(first_term)
        second_aliases = get_term_aliases(second_term)

        first_in_section = any(alias in section_text for alias in first_aliases)
        second_in_section = any(alias in section_text for alias in second_aliases)
        first_in_text = any(alias in text for alias in first_aliases)
        second_in_text = any(alias in text for alias in second_aliases)

        if not (
            (first_in_section and second_in_section)
            or (first_in_text and second_in_text)
        ):
            continue

        score = float(similarity_scores[index])

        if first_in_section and second_in_section:
            score += 0.35
        elif first_in_text and second_in_text:
            score += 0.08

        section_normalized = normalize_for_match(chunk.get("section", ""))
        implementation_hits = sum(
            normalize_for_match(term_name) in section_normalized
            for term_name in IMPLEMENTATION_SECTION_TERMS
        )
        score -= 0.20 * implementation_hits

        candidates.append((score, index))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def make_retrieval_record(index, similarity_scores, matched_term=None):
    # Converts an indexed chunk into the dictionary returned to the chat program.
    chunk = document_chunks[index]

    return {
        "source": chunk["source"],
        "section": chunk.get("section", "General"),
        "topic": chunk.get("topic", Path(chunk["source"]).stem),
        "text": chunk["text"],
        "score": float(similarity_scores[index]),
        "matched_term": matched_term,
    }


def retrieve_relevant_chunks(question, top_k=RAG_TOP_K, candidate_multiplier=6):
    # Retrieves section aware evidence, with special handling for comparison questions.
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    similarity_scores = chunk_embeddings @ question_embedding
    comparison_terms = extract_comparison_terms(question)

    # Comparison questions need either one complete comparison section or one
    # conceptual section for each side. They should not be padded with unrelated data.
    if comparison_terms:
        complete_index = find_complete_comparison_chunk(
            comparison_terms,
            similarity_scores,
        )

        if complete_index is not None:
            complete_record = make_retrieval_record(
                complete_index,
                similarity_scores,
                matched_term="complete comparison",
            )

            if complete_record["score"] >= RAG_MIN_SCORE:
                return [complete_record]

        retrieved = []
        used_sections = set()

        for term in comparison_terms[:2]:
            best_index = find_best_chunk_for_term(
                term,
                similarity_scores,
            )

            if best_index is None:
                continue

            record = make_retrieval_record(
                best_index,
                similarity_scores,
                matched_term=term,
            )
            section_key = (record["source"], record["section"])

            if section_key in used_sections:
                continue

            if record["score"] >= RAG_MIN_SCORE:
                retrieved.append(record)
                used_sections.add(section_key)

        return retrieved

    candidate_count = min(
        max(top_k * candidate_multiplier, 8),
        len(similarity_scores),
    )
    candidate_indices = np.argsort(similarity_scores)[-candidate_count:][::-1]

    for index in candidate_indices:
        score = float(similarity_scores[index])
        if score < RAG_MIN_SCORE:
            continue

        return [make_retrieval_record(index, similarity_scores)]

    return []


def select_relevant_sentences(question, retrieved_chunks, sentences_per_chunk=2):
    # Selects the most question-relevant sentences from ordinary prose chunks.
    sentence_records = []

    if not retrieved_chunks:
        return sentence_records

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", question)
        if len(word) > 2
    }

    intent = detect_question_intent(question)

    for chunk in retrieved_chunks:
        cleaned_chunk = clean_retrieved_text(chunk["text"])
        sentences = re.split(r"(?<=[.!?])\s+|[•●▪]\s*", cleaned_chunk)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        if not sentences:
            continue

        sentence_embeddings = embedding_model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sentence_scores = sentence_embeddings @ question_embedding

        for index, sentence in enumerate(sentences):
            normalized_sentence = sentence.lower()
            overlap_count = sum(
                word in normalized_sentence
                for word in question_words
            )
            sentence_scores[index] += 0.04 * overlap_count

            if intent == "limitations" and any(
                term in normalized_sentence
                for term in [
                    "limitation",
                    "disadvantage",
                    "drawback",
                    "ignore",
                    "sparse",
                    "high-dimensional",
                    "does not",
                    "cannot",
                    "difficulty",
                ]
            ):
                sentence_scores[index] += 0.20

            if intent == "advantages" and any(
                term in normalized_sentence
                for term in [
                    "advantage",
                    "benefit",
                    "efficient",
                    "better",
                    "improve",
                ]
            ):
                sentence_scores[index] += 0.15

        best_count = min(sentences_per_chunk, len(sentences))
        best_indices = np.argsort(sentence_scores)[-best_count:]
        final_indices = sorted(best_indices.tolist())

        selected_sentences = [
            sentences[index]
            for index in final_indices
        ]

        sentence_records.append({
            "source": chunk["source"],
            "section": chunk.get("section", "General"),
            "topic": chunk.get("topic", "General"),
            "text": " ".join(selected_sentences),
            "score": chunk["score"],
            "matched_term": chunk.get("matched_term"),
        })

    return sentence_records


# PROMPT HELPERS FOR OPTIONAL GENERATIVE MODE

def format_rag_input(question, retrieved_chunks):
    # Builds a grounded prompt used only when answer_mode='generative'.
    context_sections = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_sections.append(
            f"[Note {index} | {chunk.get('section', 'General')}]\n"
            f"{chunk['text']}"
        )

    retrieved_context = "\n\n".join(context_sections)

    return (
        "Answer the question using only the facts in the course notes below. "
        "Do not add unsupported information. If the notes do not answer the "
        "question, say: 'The course notes do not provide enough information "
        "to answer this question.'\n\n"
        "### Course Notes:\n"
        f"{retrieved_context}\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Answer:\n"
    )


def trim_rag_prompt(prompt_text, tokenizer, max_prompt_tokens=768):
    # Trims a RAG prompt to the model context budget without changing generation code.
    prompt_token_ids = tokenizer.encode(prompt_text)

    if len(prompt_token_ids) <= max_prompt_tokens:
        return prompt_text

    prompt_token_ids = prompt_token_ids[-max_prompt_tokens:]
    return tokenizer.decode(prompt_token_ids)


# ANSWER SYNTHESIS

def deduplicate_facts(facts):
    # Removes repeated facts while preserving their original order.
    unique_facts = []
    seen = set()

    for fact in facts:
        normalized = normalize_for_match(fact)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_facts.append(fact)

    return unique_facts

#This function solved the main issue of the syntaxis part of the model.
def split_into_clean_facts(text):
    # Splits retrieved prose or bullets into short reusable factual statements.
    cleaned = clean_retrieved_text(text)
    sentences = re.split(r"(?<=[.!?])\s+|[•●▪]\s*|\s*;\s*", cleaned)
    facts = []

    for sentence in sentences:
        sentence = sentence.strip(" .;-|")

        if not sentence:
            continue

        if len(sentence.split()) < 3:
            continue

        facts.append(sentence)

    return deduplicate_facts(facts)


def build_extractive_answer(question, retrieved_chunks, minimum_score=RAG_MIN_SCORE):
    # Returns cleaned retrieved evidence directly with minimal restructuring.
    supported_passages = []

    for chunk in retrieved_chunks:
        if chunk["score"] < minimum_score:
            continue

        cleaned_text = clean_retrieved_text(chunk["text"])
        if cleaned_text:
            supported_passages.append(cleaned_text)

    if not supported_passages:
        return (
            "The course notes do not provide enough information "
            "to answer this question."
        )

    return " ".join(supported_passages)


def synthesize_general_answer(passages):
    # Combines unique retrieved facts into a concise general-purpose answer.
    facts = []

    for passage in passages:
        facts.extend(split_into_clean_facts(passage["text"]))

    facts = deduplicate_facts(facts)

    if not facts:
        return (
            "The course notes do not provide enough information "
            "to answer this question."
        )

    return " ".join(fact.rstrip(".") + "." for fact in facts)


def synthesize_limitations_answer(passages):
    # Formats retrieved limitation facts into a direct limitations answer.
    facts = []

    for passage in passages:
        facts.extend(split_into_clean_facts(passage["text"]))

    facts = deduplicate_facts(facts)

    if not facts:
        return (
            "The course notes do not provide enough information "
            "about the limitations."
        )

    return "The main limitations are: " + "; ".join(facts) + "."


def synthesize_advantages_answer(passages):
    # Formats retrieved advantage facts into a direct advantages answer.
    facts = []

    for passage in passages:
        facts.extend(split_into_clean_facts(passage["text"]))

    facts = deduplicate_facts(facts)

    if not facts:
        return (
            "The course notes do not provide enough information "
            "about the advantages."
        )

    return "The main advantages are: " + "; ".join(facts) + "."


def parse_pipe_table(text):
    # Parses Markdown/pipe-delimited table rows while preserving column positions.
    rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or "|" not in line:
            continue

        cells = [cell.strip() for cell in line.split("|")]

        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if len(cells) < 2:
            continue

        # Skip Markdown separator rows such as | --- | --- |.
        if all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells):
            continue

        rows.append(cells)

    return rows


def cell_contains_term(cell, term):
    # Checks whether a table cell names a comparison term or one of its aliases.
    normalized_cell = normalize_for_match(cell)
    return any(alias in normalized_cell for alias in get_term_aliases(term))


def synthesize_comparison_table(first_term, second_term, table_rows):
    # Converts a two-concept comparison table into paired natural-language sentences.
    if not table_rows:
        return None

    first_col = None
    second_col = None
    header_index = None

    # Find the header row and the exact columns for both concepts.
    for row_index, row in enumerate(table_rows):
        for column_index, cell in enumerate(row):
            if first_col is None and cell_contains_term(cell, first_term):
                first_col = column_index
            if second_col is None and cell_contains_term(cell, second_term):
                second_col = column_index

        if first_col is not None and second_col is not None:
            header_index = row_index
            break

        first_col = None
        second_col = None

    pairs = []

    if header_index is not None:
        for row in table_rows[header_index + 1:]:
            if max(first_col, second_col) >= len(row):
                continue

            first_value = row[first_col].strip(" .")
            second_value = row[second_col].strip(" .")

            if not first_value or not second_value:
                continue

            label_candidates = [
                cell.strip(" .")
                for index, cell in enumerate(row)
                if index not in {first_col, second_col} and cell.strip(" .")
            ]
            label = label_candidates[0] if label_candidates else ""
            pairs.append((label, first_value, second_value))
    else:
        # Fallback for simple two-column tables without an explicit header row.
        for row in table_rows:
            nonempty = [cell.strip(" .") for cell in row if cell.strip(" .")]
            if len(nonempty) == 2:
                pairs.append(("", nonempty[0], nonempty[1]))

    if not pairs:
        return None

    sentences = []
    first_name = display_term(first_term)
    second_name = display_term(second_term)

    for label, first_value, second_value in pairs[:5]:
        label_prefix = f"For {label.lower()}, " if label else ""
        sentences.append(
            f"{label_prefix}{first_name} {first_value[0].lower() + first_value[1:] if first_value else first_value}, "
            f"while {second_name} {second_value[0].lower() + second_value[1:] if second_value else second_value}."
        )

    return " ".join(sentences)


def assign_fact_to_attribute(fact):
    # Assigns a prose fact to a broad comparison attribute such as output or loss.
    normalized_fact = fact.lower()
    best_attribute = None
    best_match_count = 0

    for attribute, keywords in COMPARISON_ATTRIBUTES.items():
        match_count = sum(keyword in normalized_fact for keyword in keywords)

        if match_count > best_match_count:
            best_match_count = match_count
            best_attribute = attribute

    return best_attribute


def fact_mentions_term(fact, term):
    # Checks whether a fact explicitly mentions one side of a comparison.
    normalized_fact = normalize_for_match(fact)
    return any(alias in normalized_fact for alias in get_term_aliases(term))


def extract_topic_facts(topic, passages):
    # Collects and categorizes facts that belong to one side of a comparison.
    topic_facts = {}

    for passage in passages:
        section = passage.get("section", "")
        section_matches = any(
            alias in normalize_for_match(section)
            for alias in get_term_aliases(topic)
        )

        for fact in split_into_clean_facts(passage["text"]):
            if not section_matches and not fact_mentions_term(fact, topic):
                continue

            attribute = assign_fact_to_attribute(fact)
            if attribute is None:
                attribute = "other"

            topic_facts.setdefault(attribute, []).append(fact)

    return topic_facts


def choose_best_fact(facts):
    # Chooses the shortest clear fact from a list of candidate facts.
    if not facts:
        return None

    return min(
        deduplicate_facts(facts),
        key=lambda fact: (len(fact.split()), len(fact)),
    )


def strip_leading_topic_name(fact, topic):
    # Removes a repeated topic name from the start of a fact for smoother comparisons.
    result = fact.strip()

    aliases = sorted(
        get_term_aliases(topic),
        key=len,
        reverse=True,
    )

    for alias in aliases:
        pattern = re.compile(
            rf"^\s*{re.escape(alias)}\s*(?:[:\-–—]|\bis\b|\buses\b|\bpredicts\b)?\s*",
            flags=re.IGNORECASE,
        )
        updated = pattern.sub("", result, count=1).strip()
        if updated != result:
            return updated

    return result


def synthesize_comparison_answer(question, passages):
    # Synthesizes comparison answers from tables first, then paired prose facts as fallback.
    comparison_terms = extract_comparison_terms(question)

    if len(comparison_terms) < 2:
        return synthesize_general_answer(passages)

    first_term, second_term = comparison_terms[:2]

    # Preserve and parse table structure before any prose flattening.
    for passage in passages:
        table_rows = parse_pipe_table(passage["text"])
        table_answer = synthesize_comparison_table(
            first_term,
            second_term,
            table_rows,
        )
        if table_answer:
            return table_answer

    first_facts = extract_topic_facts(first_term, passages)
    second_facts = extract_topic_facts(second_term, passages)

    comparison_sentences = []

    for attribute in COMPARISON_ATTRIBUTES:
        if attribute not in first_facts or attribute not in second_facts:
            continue

        first_fact = choose_best_fact(first_facts[attribute])
        second_fact = choose_best_fact(second_facts[attribute])

        if not first_fact or not second_fact:
            continue

        first_value = strip_leading_topic_name(first_fact, first_term).rstrip(".")
        second_value = strip_leading_topic_name(second_fact, second_term).rstrip(".")

        if not first_value or not second_value:
            continue

        comparison_sentences.append(
            f"{display_term(first_term)} {first_value[0].lower() + first_value[1:]}, "
            f"while {display_term(second_term)} {second_value[0].lower() + second_value[1:]}."
        )

    if comparison_sentences:
        return " ".join(deduplicate_facts(comparison_sentences))

    # If attribute pairing is not possible, present the clearest facts from each side.
    first_all = []
    for facts in first_facts.values():
        best = choose_best_fact(facts)
        if best:
            first_all.append(best)

    second_all = []
    for facts in second_facts.values():
        best = choose_best_fact(facts)
        if best:
            second_all.append(best)

    if first_all and second_all:
        first_summary = " ".join(deduplicate_facts(first_all)[:3])
        second_summary = " ".join(deduplicate_facts(second_all)[:3])
        return (
            f"{display_term(first_term)}: {first_summary} "
            f"In contrast, {display_term(second_term)}: {second_summary}"
        )

    return synthesize_general_answer(passages)


def synthesize_answer(question, retrieved_chunks):
    # Routes retrieved evidence to the appropriate deterministic synthesis strategy.
    if not retrieved_chunks:
        return (
            "The course notes do not provide enough information "
            "to answer this question."
        )

    intent = detect_question_intent(question)
    passages = []

    for chunk in retrieved_chunks:
        preserve_newlines = intent == "comparison"
        cleaned_text = clean_retrieved_text(
            chunk["text"],
            preserve_newlines=preserve_newlines,
        )

        if not cleaned_text:
            continue

        passages.append({
            "section": chunk.get("section", "General"),
            "topic": chunk.get("topic", "General"),
            "text": cleaned_text,
        })

    if not passages:
        return (
            "The course notes do not provide enough information "
            "to answer this question."
        )

    if intent == "comparison":
        return synthesize_comparison_answer(question, passages)

    if intent == "limitations":
        return synthesize_limitations_answer(passages)

    if intent == "advantages":
        return synthesize_advantages_answer(passages)

    return synthesize_general_answer(passages)


# MAIN RAG ENTRY POINT

def answer_question_with_rag(
    question,
    model,
    tokenizer,
    device,
    context_size,
    top_k=RAG_TOP_K,
    max_new_tokens=96,
    answer_mode="extractive",
):
    # Runs retrieval and returns GPT-2 output.
    retrieved_chunks = retrieve_relevant_chunks(
        question,
        top_k=top_k,
    )

    intent = detect_question_intent(question)

    # Keep full section/table structure for comparisons. Ordinary questions benefit
    # from sentence-level selection because it removes unrelated neighboring facts.
    if intent != "comparison":
        retrieved_chunks = select_relevant_sentences(
            question,
            retrieved_chunks,
            sentences_per_chunk=2,
        )

    if answer_mode == "extractive":
        answer_text = build_extractive_answer(
            question,
            retrieved_chunks,
        )
        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
        }

    if answer_mode == "synthesized":
        answer_text = synthesize_answer(
            question,
            retrieved_chunks,
        )
        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
        }

    if answer_mode != "generative":
        raise ValueError(
            "answer_mode must be 'extractive', 'synthesized', or 'generative'."
        )

    # Generative mode remains available for experiments with the fine-tuned GPT-2.
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": (
                "The course notes do not provide enough information "
                "to answer this question."
            ),
            "retrieved_chunks": [],
        }

    input_text = format_rag_input(
        question,
        retrieved_chunks,
    )

    input_text = trim_rag_prompt(
        input_text,
        tokenizer,
        max_prompt_tokens=(context_size - max_new_tokens),
    )

    prompt_token_ids = text_to_token_ids(
        input_text,
        tokenizer,
    ).to(device)

    token_ids = generate(
        model=model,
        idx=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.0,
        top_k=None,
        repetition_penalty=1.1,
        eos_id=50256,
    )

    new_token_ids = token_ids[:, prompt_token_ids.shape[1]:]
    answer_text = token_ids_to_text(new_token_ids, tokenizer).strip()

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks
    }
