# 🧪 RAGLABS
### Hybrid RAG Simulator & AI Evaluator

> **Tune. Test. Measure. Repeat.**

RAGLABS is an experimental **Hybrid RAG Simulator and AI Evaluation tool** designed to make Retrieval-Augmented Generation (RAG) experimentation easier to understand, measure, and optimize.

The idea is simple:

**What is the right RAG configuration that delivers the required answer quality without unnecessary latency and token consumption?**

Instead of treating RAG as a black box, RAGLABS allows users to manually tune retrieval parameters, run experiments, and observe how those changes affect retrieval quality, answer quality, latency, and token usage.

---

## 📻 The Idea

Remember tuning an old radio?

You would slowly turn the dial until the noise disappeared and the signal became clear.

RAG optimization works surprisingly similarly.

A very low **Top-K** may fail to retrieve enough relevant context.

A very high **Top-K** may introduce additional context, noise, tokens, and latency.

The objective is not simply to maximize Top-K.

It is to find the **sweet spot between:**

### Top-K × Quality × Latency

RAGLABS turns this trade-off into something users can experiment with directly.

---

## 🎯 Product Objective

RAGLABS is designed around a simple Applied AI Product Management question:

> **What is the smallest amount of retrieved context that delivers the required answer quality at an acceptable latency and token cost?**

The simulator enables users to modify retrieval configurations and measure their impact rather than selecting RAG parameters based purely on intuition.

---

## ⚙️ What You Can Tune

RAGLABS currently supports experimentation with:

- **Retrieval Strategy**
  - Hybrid Search
  - Vector Search Only
  - BM25 Search Only

- **BM25 Weight**

- **Vector Search Weight**

- **BM25 Top-K**

- **Vector Top-K**

- **Final Top-K**

- **Chunk Size**

- **Chunk Overlap**

For Hybrid Search, BM25 and Vector retrieval results are combined using **Reciprocal Rank Fusion (RRF)**.

---

## 🔍 Hybrid Retrieval Architecture

```text
                     User Question
                           │
                 ┌─────────┴─────────┐
                 │                   │
              BM25 Search       Vector Search
                 │                   │
                 └─────────┬─────────┘
                           │
                Reciprocal Rank Fusion
                           │
                       Final Top-K
                           │
                     Retrieved Context
                           │
                           ▼
                          LLM
                           │
                    Generated Answer
                           │
                           ▼
                      AI Evaluator
```

Hybrid retrieval combines:

**BM25 Search**  
Useful for lexical and exact keyword matching.

**Vector Search**  
Useful for semantic similarity and meaning-based retrieval.

**Reciprocal Rank Fusion**  
Combines ranked results from both retrieval approaches into a final retrieval set.

---

## 📊 AI Evaluation

RAGLABS evaluates both **retrieval quality** and **generation quality**.

### Retrieval Quality

**Hit Rate@K**

Measures whether at least one relevant chunk appears in the retrieved Top-K results.

**Reciprocal Rank (RR)**

Measures how early the first relevant result appears in the ranked retrieval results.

> For a benchmark containing multiple questions, Reciprocal Rank can be averaged to calculate Mean Reciprocal Rank (MRR).

### Generation Quality

**Faithfulness**

Measures whether the generated answer is supported by the retrieved context.

**Correctness**

Measures how closely the generated answer matches the expected reference answer when ground truth is available.

**Answer Relevance**

Measures whether the response directly addresses the user's question.

---

## ⚡ Performance Metrics

RAGLABS also tracks operational metrics including:

- Retrieval latency
- Generation latency
- Evaluation latency
- Total latency
- Input tokens
- Output tokens
- Total generation tokens
- Evaluator token consumption

This enables experiments to consider both **AI quality and system performance**.

---

## 🧪 Example Experiment

Consider testing the same evaluation dataset using different Final Top-K values:

| Final Top-K | Hit Rate | RR / MRR | Faithfulness | Correctness | Input Tokens | Latency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | Measure | Measure | Measure | Measure | Measure | Measure |
| 3 | Measure | Measure | Measure | Measure | Measure | Measure |
| 5 | Measure | Measure | Measure | Measure | Measure | Measure |
| 8 | Measure | Measure | Measure | Measure | Measure | Measure |
| 10 | Measure | Measure | Measure | Measure | Measure | Measure |

The objective is not automatically to select the configuration with the highest Top-K.

Instead, the experiment asks:

> **At what point does additional retrieved context stop producing meaningful quality improvement relative to the additional latency and token consumption?**

---

## 🧠 LLM-as-a-Judge

RAGLABS includes an **LLM-as-a-Judge evaluation layer**.

The evaluator reviews the:

```text
Question
    +
Retrieved Context
    +
Generated Answer
    +
Reference Answer (when available)
```

and produces structured evaluation scores for:

- Faithfulness
- Correctness
- Answer Relevance

Evaluator token consumption is tracked separately from generation token consumption to make evaluation overhead visible.

---

## 🏗️ Current Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS |
| Lexical Retrieval | BM25 / `rank-bm25` |
| RAG Framework | LangChain |
| Fusion Strategy | Reciprocal Rank Fusion |
| Evaluation | LLM-as-a-Judge |
| Language | Python |

---

## 📂 Project Structure

```text
RAGLABS/
│
├── app.py
│
├── core/
│   ├── __init__.py
│   ├── rag_engine.py
│   └── evaluator.py
│
├── components/
│   └── __init__.py
│
├── data/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

> `.env` and the virtual environment are excluded from Git tracking.

---

## 🚀 Running RAGLABS Locally

### 1. Clone the repository

```bash
git clone https://github.com/sathiyakrishna/raglabs-hybrid-rag-evaluator.git
```

### 2. Enter the project

```bash
cd raglabs-hybrid-rag-evaluator
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows / Git Bash**

```bash
source venv/Scripts/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Create a `.env` file

```text
OPENAI_API_KEY=your_openai_api_key
```

Never commit API keys or `.env` files to GitHub.

### 7. Start RAGLABS

```bash
streamlit run app.py
```

---

## 🔬 How to Experiment

1. Upload one or more PDF knowledge-base documents.
2. Select the retrieval strategy.
3. Adjust BM25 and Vector weights.
4. Configure retrieval Top-K values.
5. Configure chunk size and overlap.
6. Ask a question.
7. Run the experiment.
8. Review retrieved chunks.
9. Compare retrieval and generation quality.
10. Observe latency and token consumption.
11. Change one parameter and repeat.

The goal is to make RAG optimization an **observable experiment rather than guesswork**.

---

## 🗺️ Roadmap

RAGLABS is currently an experimental project.

Planned improvements include:

- Retrieval/index caching
- Experiment history
- Side-by-side experiment comparison
- Automated Top-K sweeps
- Top-K vs Quality visualization
- Top-K vs Latency visualization
- Token-cost analysis
- Benchmark evaluation across multiple questions
- MRR and additional retrieval metrics
- RAG configuration comparison
- Exportable evaluation reports

The longer-term goal is to help identify an effective **quality × latency × cost operating point** for a RAG system.

---

## 💡 Applied AI Product Management Perspective

Building a RAG pipeline is only part of the problem.

From an Applied AI Product Management perspective, the more important questions are:

**How do we know retrieval is working?**

**How do we know the generated answer is grounded and correct?**

**What happens to quality when retrieval parameters change?**

**How much latency and token cost are we accepting for an incremental improvement in quality?**

RAGLABS was built to explore those questions through measurable experiments.

---

## 👤 Author

### Satya Krishna

**Applied AI Product Manager | AI Product Management | Finance Transformation**

GitHub:  
https://github.com/sathiyakrishna

LinkedIn:  
https://www.linkedin.com/in/sathiyakrishna/

---

## ⭐ RAGLABS

**Tune the retrieval. Measure the quality. Understand the trade-off.**

**Tune. Test. Measure. Repeat. 📻 → 🤖**
