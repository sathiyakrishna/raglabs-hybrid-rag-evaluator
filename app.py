# ============================================================
# RAGLABS
# Hybrid RAG Simulator & AI Evaluator
# ============================================================

import time
import streamlit as st

from core.rag_engine import (
    process_pdfs,
    build_retrievers,
    hybrid_retrieve,
    generate_answer,
)

from core.evaluator import evaluate_rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAGLABS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .raglabs-title {
        font-size: 2.7rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .raglabs-subtitle {
        font-size: 1.05rem;
        color: #6B7280;
        margin-top: 0.25rem;
        margin-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        padding: 16px;
        border-radius: 12px;
        background-color: #FFFFFF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="raglabs-title">RAGLABS</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="raglabs-subtitle">
    Hybrid RAG Simulator & AI Evaluator
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("RAG Configuration")

    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        [
            "Hybrid",
            "Vector Only",
            "BM25 Only",
        ],
    )

    st.subheader("Hybrid Retrieval")

    bm25_weight = st.slider(
        "BM25 Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
    )

    vector_weight = 1.0 - bm25_weight

    st.metric(
        "Vector Weight",
        f"{vector_weight:.1f}",
    )

    bm25_k = st.slider(
        "BM25 Top-K",
        min_value=1,
        max_value=20,
        value=5,
    )

    vector_k = st.slider(
        "Vector Top-K",
        min_value=1,
        max_value=20,
        value=5,
    )

    final_k = st.slider(
        "Final Top-K",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.subheader("Chunking")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=500,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=25,
    )

    if chunk_overlap >= chunk_size:
        st.warning(
            "Chunk overlap must be smaller than chunk size."
        )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

st.subheader("Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:

    st.caption(
        f"{len(uploaded_files)} document(s) selected"
    )


st.divider()


# ============================================================
# ASK RAGLABS
# ============================================================

st.subheader("Ask RAGLABS")

question = st.text_area(
    "Question",
    placeholder="Ask a question about your uploaded documents...",
    height=100,
)

run = st.button(
    "Run Experiment",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

if run:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif chunk_overlap >= chunk_size:

        st.warning(
            "Chunk overlap must be smaller than chunk size."
        )

    else:

        try:

            total_start = time.perf_counter()

            # =================================================
            # PHASE 1 — DOCUMENT PROCESSING
            # =================================================

            with st.spinner(
                "Processing knowledge base..."
            ):

                chunks = process_pdfs(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )


            # =================================================
            # PHASE 2 — BUILD RETRIEVERS
            # =================================================

            with st.spinner(
                "Building BM25 and vector indexes..."
            ):

                bm25_retriever, vector_retriever = (
                    build_retrievers(
                        chunks,
                        bm25_k=bm25_k,
                        vector_k=vector_k,
                    )
                )


            # =================================================
            # RETRIEVAL STRATEGY
            # =================================================

            if retrieval_mode == "Hybrid":

                active_bm25_weight = bm25_weight
                active_vector_weight = vector_weight

            elif retrieval_mode == "Vector Only":

                active_bm25_weight = 0.0
                active_vector_weight = 1.0

            else:

                active_bm25_weight = 1.0
                active_vector_weight = 0.0


            # =================================================
            # PHASE 3 — RETRIEVAL
            # =================================================

            retrieval_start = time.perf_counter()

            with st.spinner(
                "Retrieving relevant context..."
            ):

                retrieved_results = hybrid_retrieve(
                    query=question,
                    bm25_retriever=bm25_retriever,
                    vector_retriever=vector_retriever,
                    bm25_weight=active_bm25_weight,
                    vector_weight=active_vector_weight,
                    final_k=final_k,
                )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )


            # =================================================
            # PHASE 4 — GENERATION
            # =================================================

            generation_start = time.perf_counter()

            with st.spinner(
                "Generating answer..."
            ):

                rag_result = generate_answer(
                    query=question,
                    retrieved_results=retrieved_results,
                )

            generation_latency = (
                time.perf_counter()
                - generation_start
            )


            # =================================================
            # PHASE 5 — AI EVALUATION
            # =================================================

            evaluation_start = time.perf_counter()

            with st.spinner(
                "Evaluating RAG quality..."
            ):

                evaluation = evaluate_rag(
                    question=question,
                    generated_answer=rag_result["answer"],
                    retrieved_results=retrieved_results,
                )

            evaluation_latency = (
                time.perf_counter()
                - evaluation_start
            )


            # =================================================
            # TOTAL LATENCY
            # =================================================

            total_latency = (
                time.perf_counter()
                - total_start
            )


            # =================================================
            # SAVE RESULTS
            # =================================================

            st.session_state["rag_result"] = rag_result

            st.session_state[
                "retrieved_results"
            ] = retrieved_results

            st.session_state[
                "evaluation"
            ] = evaluation

            st.session_state[
                "retrieval_latency"
            ] = retrieval_latency

            st.session_state[
                "generation_latency"
            ] = generation_latency

            st.session_state[
                "evaluation_latency"
            ] = evaluation_latency

            st.session_state[
                "total_latency"
            ] = total_latency

            st.session_state[
                "retrieval_mode"
            ] = retrieval_mode

            st.session_state[
                "active_bm25_weight"
            ] = active_bm25_weight

            st.session_state[
                "active_vector_weight"
            ] = active_vector_weight

            st.session_state[
                "chunk_count"
            ] = len(chunks)


        except Exception as e:

            st.error(
                f"Experiment failed: {e}"
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "rag_result" in st.session_state:

    rag_result = st.session_state[
        "rag_result"
    ]

    evaluation = st.session_state.get(
        "evaluation",
        {},
    )

    retrieved_results = st.session_state.get(
        "retrieved_results",
        [],
    )


    # ========================================================
    # GENERATED ANSWER
    # ========================================================

    st.divider()

    st.subheader(
        "Generated Answer"
    )

    st.success(
        rag_result.get(
            "answer",
            "No answer generated.",
        )
    )


    # ========================================================
    # CONFIGURATION SUMMARY
    # ========================================================

    st.subheader(
        "Experiment Configuration"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Retrieval",
            st.session_state.get(
                "retrieval_mode",
                "—",
            ),
        )

    with c2:

        st.metric(
            "BM25 Weight",
            f"{st.session_state.get('active_bm25_weight', 0):.1f}",
        )

    with c3:

        st.metric(
            "Vector Weight",
            f"{st.session_state.get('active_vector_weight', 0):.1f}",
        )

    with c4:

        st.metric(
            "Chunks",
            st.session_state.get(
                "chunk_count",
                "—",
            ),
        )


    # ========================================================
    # RETRIEVED CONTEXT
    # ========================================================

    st.subheader(
        "Retrieved Context"
    )

    for index, item in enumerate(
        retrieved_results,
        start=1,
    ):

        doc = item["document"]

        fusion_score = item[
            "fusion_score"
        ]

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page = doc.metadata.get(
            "page"
        )

        page_display = (
            page + 1
            if isinstance(page, int)
            else "Unknown"
        )

        with st.expander(
            f"Rank {index} • "
            f"{source} • "
            f"Page {page_display}"
        ):

            st.write(
                doc.page_content
            )

            st.caption(
                f"Fusion Score: "
                f"{fusion_score:.6f}"
            )


    # ========================================================
    # AI EVALUATION
    # ========================================================

    st.divider()

    st.subheader(
        "AI Evaluation"
    )


    # --------------------------------------------------------
    # RETRIEVAL QUALITY
    # --------------------------------------------------------

    st.caption(
        "Retrieval Quality"
    )

    r1, r2 = st.columns(2)

    hit_rate = evaluation.get(
        "hit_rate"
    )

    reciprocal_rank = evaluation.get(
        "mrr"
    )


    with r1:

        st.metric(
            "Hit Rate@K",
            (
                f"{hit_rate:.2f}"
                if hit_rate is not None
                else "N/A"
            ),
        )


    with r2:

        st.metric(
            "Reciprocal Rank",
            (
                f"{reciprocal_rank:.2f}"
                if reciprocal_rank is not None
                else "N/A"
            ),
        )


    # --------------------------------------------------------
    # GENERATION QUALITY
    # --------------------------------------------------------

    st.caption(
        "Generation Quality"
    )

    g1, g2, g3 = st.columns(3)

    faithfulness = evaluation.get(
        "faithfulness"
    )

    correctness = evaluation.get(
        "correctness"
    )

    answer_relevance = evaluation.get(
        "answer_relevance"
    )


    with g1:

        st.metric(
            "Faithfulness",
            (
                f"{faithfulness:.2f}"
                if faithfulness is not None
                else "N/A"
            ),
        )


    with g2:

        st.metric(
            "Correctness",
            (
                f"{correctness:.2f}"
                if correctness is not None
                else "N/A"
            ),
        )


    with g3:

        st.metric(
            "Answer Relevance",
            (
                f"{answer_relevance:.2f}"
                if answer_relevance is not None
                else "N/A"
            ),
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.caption(
        "RAG Performance"
    )

    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "Input Tokens",
            rag_result.get(
                "input_tokens",
                0,
            ),
        )


    with p2:

        st.metric(
            "Output Tokens",
            rag_result.get(
                "output_tokens",
                0,
            ),
        )


    with p3:

        st.metric(
            "Total Tokens",
            rag_result.get(
                "total_tokens",
                0,
            ),
        )


    with p4:

        st.metric(
            "Total Latency",
            f"{st.session_state.get('total_latency', 0):.2f}s",
        )


    # ========================================================
    # LATENCY BREAKDOWN
    # ========================================================

    with st.expander(
        "Latency Breakdown"
    ):

        l1, l2, l3 = st.columns(3)

        with l1:

            st.metric(
                "Retrieval",
                f"{st.session_state.get('retrieval_latency', 0):.2f}s",
            )

        with l2:

            st.metric(
                "Generation",
                f"{st.session_state.get('generation_latency', 0):.2f}s",
            )

        with l3:

            st.metric(
                "Evaluation",
                f"{st.session_state.get('evaluation_latency', 0):.2f}s",
            )


    # ========================================================
    # EVALUATOR COST / TOKENS
    # ========================================================

    with st.expander(
        "Evaluator Usage"
    ):

        e1, e2, e3 = st.columns(3)

        with e1:

            st.metric(
                "Evaluator Input Tokens",
                evaluation.get(
                    "evaluator_input_tokens",
                    0,
                ),
            )

        with e2:

            st.metric(
                "Evaluator Output Tokens",
                evaluation.get(
                    "evaluator_output_tokens",
                    0,
                ),
            )

        with e3:

            st.metric(
                "Evaluator Total Tokens",
                evaluation.get(
                    "evaluator_total_tokens",
                    0,
                ),
            )


    # ========================================================
    # EVALUATION EXPLANATION
    # ========================================================

    with st.expander(
        "Why did RAGLABS give these scores?"
    ):

        if evaluation.get(
            "reference_answer"
        ):

            st.markdown(
                "**Reference Answer**"
            )

            st.write(
                evaluation[
                    "reference_answer"
                ]
            )

        else:

            st.info(
                "No ground-truth answer is available "
                "for this question. Retrieval Hit Rate "
                "and Reciprocal Rank cannot be calculated "
                "reliably."
            )


        st.markdown(
            "**Faithfulness**"
        )

        st.write(
            evaluation.get(
                "faithfulness_reason",
                "No explanation available.",
            )
        )


        st.markdown(
            "**Correctness**"
        )

        st.write(
            evaluation.get(
                "correctness_reason",
                "No explanation available.",
            )
        )


        st.markdown(
            "**Answer Relevance**"
        )

        st.write(
            evaluation.get(
                "answer_relevance_reason",
                "No explanation available.",
            )
        )


# ============================================================
# INITIAL EMPTY EVALUATION STATE
# ============================================================

else:

    st.divider()

    st.subheader(
        "AI Evaluation"
    )

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.metric("Hit Rate@K", "—")

    with e2:
        st.metric("Reciprocal Rank", "—")

    with e3:
        st.metric("Faithfulness", "—")

    with e4:
        st.metric("Correctness", "—")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RAGLABS • Hybrid RAG Simulator & AI Evaluator • "
    "BM25 + Vector Search + Reciprocal Rank Fusion"
)