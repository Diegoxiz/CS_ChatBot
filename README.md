# CS_ChatBot

CS_ChatBot is a chatbot designed to assist Computer Science students with concepts from different courses throughout their degree. The chatbot contains information from multiple CS-related subjects, including **Linear Algebra, Natural Language Processing, Software Engineering, and more**.

The project combines **Natural Language Processing (NLP), Machine Learning, Large Language Models (LLMs), and Retrieval-Augmented Generation (RAG)** to retrieve relevant academic information and produce answers to student questions.

Rather than relying exclusively on the pretrained knowledge of a language model, CS_ChatBot uses a custom knowledge base constructed from academic material and course notes.

# Motivation

I became interested in Machine Learning and Natural Language Processing during my Computer Science studies and wanted to apply what I had learned to a practical project.

At the same time, I wanted to build something that could be useful to other Computer Science students. This led to the idea of creating a course-oriented chatbot that students could use as an additional study resource when reviewing concepts, asking questions, or studying material from different CS courses.

The project has also served as an opportunity for me to gain practical experience with language models, model fine-tuning, semantic search, embeddings, RAG systems, and answer synthesis.

# Features

The current version of CS_ChatBot supports:

* Question answering across multiple Computer Science subjects
* A custom academic knowledge base
* Semantic retrieval using vector embeddings
* Section-aware document retrieval
* Intent detection
* Comparison-aware retrieval
* Extractive answer generation
* Synthesized answer generation
* Generative answer mode
* Source reporting
* Out-of-domain question rejection
* Retrieval of information from multiple course documents

# System Architecture

The primary RAG pipeline follows this general architecture:

```text
User Question
      ↓
Intent Detection
      ↓
Embedding / Semantic Search
      ↓
Section-Aware Retrieval
      ↓
Relevant Course Notes
      ↓
Answer Synthesis
      ↓
Grounded Answer + Sources
```

When a user submits a question, the system first analyzes the query and determines the type of information being requested.

The question is converted into an embedding and compared against embeddings generated from the documents in the knowledge base. The system then retrieves the sections that are most semantically relevant to the question.

Relevant information from these sections is processed by the answer synthesis system, which attempts to construct a concise response grounded in the retrieved course material.

The sources used to construct the answer are then displayed alongside their similarity scores.

# Base Language Model / Model Architecture

The language model used by this project is based on **GPT-2 Medium**, a Transformer-based autoregressive language model.

GPT-2 uses the Transformer architecture and generates text by predicting the next token based on the tokens that precede it. The Transformer architecture relies heavily on the attention mechanism, which allows the model to determine which parts of the input sequence are most relevant when processing and generating text.

For this project, the pretrained GPT-2 Medium model was adapted for question-answering behavior using a custom dataset containing Computer Science questions and answers.

The project preserves a generative mode that allows the fine-tuned language model to generate answers directly. However, the current recommended mode uses the RAG and answer-synthesis pipeline because testing showed that unrestricted generation could occasionally produce plausible but unsupported information.

# Fine-Tuning

The pretrained language model was fine-tuned using a custom question-answer dataset created from Computer Science course material.

The fine-tuning dataset contains questions and corresponding answers covering subjects such as Natural Language Processing, Linear Algebra, and Software Engineering.

During training, the model learns to adapt its pretrained language-generation capabilities toward the structure and terminology commonly found in Computer Science questions.

Model checkpoints are saved after training so that the chatbot can load the fine-tuned model directly without requiring the training process to be repeated every time the program is executed.

Training and validation loss were also recorded during development to evaluate the training process and preserve experimental results for later comparison.

# Dataset and Knowledge Base

The project's knowledge base was constructed primarily from academic notes and course material that I collected throughout my Computer Science studies.

The original material was cleaned and converted into plain-text documents so that it could be processed by the retrieval system.

The current knowledge base includes material from areas such as:

* Natural Language Processing
* Linear Algebra
* Software Engineering

The project uses this material in two different ways.

First, structured question-answer records are used during the **fine-tuning process** to adapt the language model.

Second, cleaned text documents are used by the **RAG system** as an external knowledge base. This allows the chatbot to retrieve relevant information at query time instead of depending entirely on information encoded in the language model's parameters.

# Retrieval-Augmented Generation (RAG) System

The RAG system was introduced to improve the accuracy and reliability of the chatbot's answers.

Instead of asking the language model to answer every question entirely from its internal parameters, the system searches the custom knowledge base for information related to the user's question.

The retrieval pipeline performs several steps:

1. Documents are loaded from the knowledge base.
2. Documents are divided into meaningful sections and chunks.
3. Text chunks are converted into numerical embeddings.
4. The user's question is converted into an embedding using the same embedding model.
5. Similarity scores are calculated between the question and knowledge-base chunks.
6. The most relevant sections are selected.
7. Duplicate or low-quality retrieval results are filtered when possible.
8. Relevant information is passed to the answer-synthesis system.
9. The final answer is returned together with its retrieved sources.

The system also includes **section-aware retrieval**, which attempts to preserve the logical organization of the original notes instead of treating every document as an arbitrary collection of text chunks.

Comparison-aware retrieval was also introduced to improve questions that require information about multiple concepts.

# Answer Synthesis

Early versions of the project relied more heavily on the fine-tuned language model to generate answers from retrieved context.

Testing showed that the language model could sometimes produce answers that sounded reasonable but contained incorrect or unsupported information. Because of this, a more controlled answer-synthesis system was introduced.

The current synthesized mode attempts to construct answers directly from information retrieved from the knowledge base.

Different types of questions can be processed differently. For example, definition questions can retrieve concise explanatory statements, while questions asking for limitations or advantages can combine several relevant facts into a single answer.

Comparison questions use additional logic intended to identify the concepts being compared and align information associated with each concept.

The project currently supports three answer modes:

* **Extractive:** Returns information directly extracted from relevant course material.
* **Synthesized:** Organizes retrieved information into a cleaner and more direct response. This is the recommended mode for the current version.
* **Generative:** Uses the fine-tuned GPT-2 model to generate an answer from retrieved context.

This design makes it possible to experiment with the trade-off between controlled, grounded answers and more flexible language-model generation.

# Testing and Evaluation

The system has been tested manually using questions covering definitions, explanations, limitations, comparisons, and topics outside the knowledge base.

Examples include:

```text
What is Bag-of-Words?

What are the limitations of Bag-of-Words?

How are embeddings learned in Word2Vec?

Compare linear regression and logistic regression.

Compare CBOW and Skip-Gram.

What is the difference between RNNs and LSTMs?

What is the capital of France?
```

The current version generally performs well on direct questions asking **what something is, why something occurs, or how a concept works**.

For example, the system successfully retrieves relevant sections for questions about Bag-of-Words, Word2Vec, logistic regression, and other concepts represented in the knowledge base.

Out-of-domain testing is also performed. When the system cannot find sufficiently relevant information in the course material, it is designed to return a response such as:

> The course notes do not provide enough information to answer this question.

This behavior helps reduce the likelihood of generating unsupported answers.

Testing results from different stages of development are preserved separately so that future versions of the system can be compared against previous implementations.

# Known Limitations

The current version performs best on direct explanatory questions, particularly questions structured around **"what," "why," and "how."**

However, comparison questions remain one of the primary weaknesses of the current system.

Questions that require comparing two or more concepts can require information to be retrieved from multiple sections or documents. The system must then determine which facts belong to each concept and align those facts correctly before constructing the final answer.

When the knowledge base contains a clearly structured comparison section, the system can often produce a useful comparison. However, when information about the concepts is distributed across multiple sections, retrieval and synthesis can become less reliable.

Comparison answers may therefore contain incomplete information, awkward sentence structure, or occasionally retrieve a section that mentions both concepts without actually explaining their differences.

For example, during testing, comparisons between Linear Regression and Logistic Regression produced useful results, while comparisons between RNNs and LSTMs were less reliable because the retrieved section mentioned both architectures without containing enough information to properly distinguish them.

Improving multi-section retrieval, comparison ranking, and comparison synthesis is planned for future versions.

Another limitation is that the quality of the chatbot depends heavily on the information available in the custom knowledge base. If a subject is missing or insufficiently documented, the system may not be able to provide an answer.

# Future Improvements

Future versions of CS_ChatBot may focus on:

* Improving comparison-aware retrieval
* Improving comparison answer synthesis
* Better multi-section information aggregation
* Retrieval reranking
* More robust intent detection
* Expanding the Computer Science knowledge base
* Improving the quality and size of the fine-tuning dataset
* Additional automated evaluation methods
* Reducing awkward wording produced by deterministic synthesis
* Experimenting with different language models and embedding models
* Improving hallucination detection and answer grounding

The current version should be considered the first functional baseline of the project. Future versions will build upon the testing results and limitations discovered during development.

# Version

**Current Version: v1.0**

Version 1.0 represents the first functional release of CS_ChatBot, including the fine-tuned language model, custom Computer Science knowledge base, semantic retrieval system, section-aware RAG pipeline, multiple answer modes, answer synthesis, source reporting, and out-of-domain rejection.
