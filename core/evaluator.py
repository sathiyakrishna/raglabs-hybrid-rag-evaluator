# ============================================================
# RAGLABS
# AI Evaluation Engine
# ============================================================
#
# Evaluation types:
#
# 1. Retrieval Evaluation
#    - Hit Rate@K
#    - Mean Reciprocal Rank (MRR)
#
# 2. Generation Evaluation
#    - Faithfulness
#    - Correctness
#    - Answer Relevance
#
# Faithfulness and relevance use LLM-as-a-Judge.
# Correctness compares the generated answer against a
# reference / ground-truth answer.
#
# ============================================================

import json
import re

from langchain_openai import ChatOpenAI


# ============================================================
# GOLDEN EVALUATION DATASET
# ============================================================
#
# This is our initial controlled evaluation dataset for the
# fictional NovaTech PDF.
#
# Later we can move this into:
#
# data/evaluation_dataset.json
#
# so users can upload their own evaluation datasets.
# ============================================================

GOLDEN_DATASET = {

    "what is the enterprise severity 1 initial response target?": {
        "reference_answer": "30 minutes, 24×7.",
        "relevant_keywords": [
            "enterprise",
            "severity 1",
            "30-minute",
            "30 minute",
            "24×7",
        ],
    },

    "how long is a deleted ticket recoverable?": {
        "reference_answer": "30 days.",
        "relevant_keywords": [
            "deleted",
            "ticket",
            "recoverable",
            "30 days",
        ],
    },

    "what is the default professional api rate limit?": {
        "reference_answer": (
            "300 requests per minute per organization."
        ),
        "relevant_keywords": [
            "professional",
            "300 requests",
            "per minute",
            "api",
        ],
    },

    "does the 14-day trial require a credit card?": {
        "reference_answer": (
            "No. The Professional trial does not require "
            "a credit card."
        ),
        "relevant_keywords": [
            "14-day",
            "trial",
            "credit card",
        ],
    },

    (
        "what is the annual professional price "
        "per named user per month?"
    ): {
        "reference_answer": (
            "₹1,299 per named user per month when billed annually."
        ),
        "relevant_keywords": [
            "professional",
            "1,299",
            "annually",
            "named user",
        ],
    },

    (
        "how long are enterprise audit logs "
        "retained by default?"
    ): {
        "reference_answer": "365 days.",
        "relevant_keywords": [
            "enterprise",
            "audit logs",
            "365 days",
        ],
    },

    (
        "what are the enterprise disaster recovery "
        "design targets?"
    ): {
        "reference_answer": (
            "RPO of 4 hours and RTO of 8 hours."
        ),
        "relevant_keywords": [
            "rpo",
            "4 hours",
            "rto",
            "8 hours",
        ],
    },

    "what happens after a failed payment?": {
        "reference_answer": (
            "The account enters a 7-day grace period. "
            "If payment remains unresolved, access may "
            "be restricted to administrators."
        ),
        "relevant_keywords": [
            "failed payment",
            "7-day",
            "grace period",
            "administrators",
        ],
    },

    "which plans support outbound webhooks?": {
        "reference_answer": (
            "Professional and Enterprise."
        ),
        "relevant_keywords": [
            "professional",
            "enterprise",
            "outbound webhooks",
        ],
    },

    (
        "what metrics are recommended for "
        "monitoring ai quality?"
    ): {
        "reference_answer": (
            "Groundedness or faithfulness, answer relevance, "
            "correctness, Hit Rate, MRR, Recall@K, latency, "
            "and token consumption."
        ),
        "relevant_keywords": [
            "faithfulness",
            "answer relevance",
            "correctness",
            "hit rate",
            "mrr",
            "recall",
            "latency",
            "token",
        ],
    },

    (
        "can an annual subscription cancellation "
        "automatically receive a prorated refund?"
    ): {
        "reference_answer": (
            "No. Early cancellation does not automatically "
            "create a prorated refund."
        ),
        "relevant_keywords": [
            "annual",
            "cancellation",
            "prorated refund",
        ],
    },

    (
        "what encryption standard is used "
        "for data at rest?"
    ): {
        "reference_answer": (
            "AES-256 or an equivalent cloud-provider "
            "managed encryption standard."
        ),
        "relevant_keywords": [
            "aes-256",
            "encrypted at rest",
            "encryption",
        ],
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for question matching.
    """

    if not text:
        return ""

    text = text.strip().lower()

    # Collapse multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# GET GOLDEN RECORD
# ============================================================

def get_golden_record(question):
    """
    Find the ground-truth evaluation record for a question.

    Returns None when the question is not part of the
    controlled golden dataset.
    """

    normalized_question = normalize_text(question)

    return GOLDEN_DATASET.get(
        normalized_question
    )


# ============================================================
# DETERMINE WHETHER A CHUNK IS RELEVANT
# ============================================================

def is_relevant_chunk(
    chunk_text,
    relevant_keywords,
):
    """
    Determine whether a retrieved chunk contains enough
    ground-truth evidence to be considered relevant.

    For our controlled simulator, a chunk is considered
    relevant when it contains at least two expected
    evidence keywords.

    Later, this can be replaced with manually labelled
    relevant chunk IDs.
    """

    if not chunk_text:
        return False

    chunk_text = chunk_text.lower()

    matches = 0

    for keyword in relevant_keywords:

        if keyword.lower() in chunk_text:
            matches += 1

    # Require at least two evidence signals.
    #
    # If the golden record only contains one keyword,
    # require that one keyword.
    required_matches = min(
        2,
        len(relevant_keywords),
    )

    return matches >= required_matches


# ============================================================
# HIT RATE @ K
# ============================================================

def calculate_hit_rate(
    retrieved_results,
    relevant_keywords,
):
    """
    Hit Rate@K:

    1.0 = at least one relevant document appears
          in the retrieved Top-K results.

    0.0 = no relevant document appears.
    """

    for item in retrieved_results:

        doc = item["document"]

        if is_relevant_chunk(
            doc.page_content,
            relevant_keywords,
        ):
            return 1.0

    return 0.0


# ============================================================
# RECIPROCAL RANK
# ============================================================

def calculate_reciprocal_rank(
    retrieved_results,
    relevant_keywords,
):
    """
    Reciprocal Rank:

        RR = 1 / rank_of_first_relevant_result

    Example:

    Relevant result at Rank 1 -> 1.00
    Relevant result at Rank 2 -> 0.50
    Relevant result at Rank 3 -> 0.33

    For one query this is Reciprocal Rank (RR).

    MRR is the mean of RR values across multiple queries.
    In the current single-question UI, we display this
    per-query RR under the MRR label for simplicity.
    """

    for rank, item in enumerate(
        retrieved_results,
        start=1,
    ):

        doc = item["document"]

        if is_relevant_chunk(
            doc.page_content,
            relevant_keywords,
        ):

            return 1.0 / rank

    return 0.0


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(retrieved_results):
    """
    Combine retrieved chunks into evaluation context.
    """

    context_parts = []

    for index, item in enumerate(
        retrieved_results,
        start=1,
    ):

        doc = item["document"]

        context_parts.append(
            f"[Chunk {index}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SAFE JSON EXTRACTION
# ============================================================

def parse_json_response(content):
    """
    Safely extract JSON from an LLM response.
    """

    if not content:
        raise ValueError(
            "Evaluator returned an empty response."
        )

    content = content.strip()

    # Remove Markdown code fences if the model
    # unexpectedly returns them.
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"^```\s*",
        "",
        content,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    return json.loads(
        content.strip()
    )


# ============================================================
# CLAMP SCORE
# ============================================================

def clamp_score(value):
    """
    Ensure evaluator scores remain between 0 and 1.
    """

    try:

        value = float(value)

    except (TypeError, ValueError):

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


# ============================================================
# LLM-AS-A-JUDGE
# ============================================================

def evaluate_generation(
    question,
    generated_answer,
    retrieved_results,
    reference_answer=None,
):
    """
    Evaluate generation quality using an LLM judge.

    Returns:

    - Faithfulness
    - Correctness
    - Answer Relevance
    - Explanations
    - Evaluator token usage
    """

    context = build_context(
        retrieved_results
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    reference_text = (
        reference_answer
        if reference_answer
        else "No reference answer is available."
    )

    prompt = f"""
You are the evaluation engine for RAGLABS.

Evaluate a Retrieval-Augmented Generation response.

You must score the answer using ONLY the definitions below.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_text}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}


METRIC 1 — FAITHFULNESS

Measure whether claims in the generated answer are supported
by the retrieved context.

Score:

1.0 = fully supported by retrieved context
0.75 = mostly supported with a minor unsupported statement
0.50 = partially supported
0.25 = mostly unsupported
0.0 = contradicted or unsupported


METRIC 2 — CORRECTNESS

Compare the generated answer with the reference answer.

Score:

1.0 = fully correct
0.75 = mostly correct with a minor omission
0.50 = partially correct
0.25 = mostly incorrect
0.0 = incorrect or contradictory

If no reference answer is available, return null for
correctness.


METRIC 3 — ANSWER RELEVANCE

Measure how directly the generated answer addresses
the user's question.

Score:

1.0 = directly and completely answers the question
0.75 = relevant but slightly incomplete
0.50 = partially relevant
0.25 = weakly related
0.0 = irrelevant


Return ONLY valid JSON.

Use exactly this structure:

{{
    "faithfulness": 0.0,
    "correctness": 0.0,
    "answer_relevance": 0.0,
    "faithfulness_reason": "short explanation",
    "correctness_reason": "short explanation",
    "answer_relevance_reason": "short explanation"
}}
"""

    response = llm.invoke(
        prompt
    )

    parsed = parse_json_response(
        response.content
    )

    faithfulness = clamp_score(
        parsed.get(
            "faithfulness",
            0,
        )
    )

    answer_relevance = clamp_score(
        parsed.get(
            "answer_relevance",
            0,
        )
    )

    raw_correctness = parsed.get(
        "correctness"
    )

    if (
        reference_answer is None
        or raw_correctness is None
    ):

        correctness = None

    else:

        correctness = clamp_score(
            raw_correctness
        )

    usage = (
        response.usage_metadata
        or {}
    )

    evaluator_input_tokens = usage.get(
        "input_tokens",
        0,
    )

    evaluator_output_tokens = usage.get(
        "output_tokens",
        0,
    )

    return {

        "faithfulness":
            faithfulness,

        "correctness":
            correctness,

        "answer_relevance":
            answer_relevance,

        "faithfulness_reason":
            parsed.get(
                "faithfulness_reason",
                "",
            ),

        "correctness_reason":
            parsed.get(
                "correctness_reason",
                "",
            ),

        "answer_relevance_reason":
            parsed.get(
                "answer_relevance_reason",
                "",
            ),

        "evaluator_input_tokens":
            evaluator_input_tokens,

        "evaluator_output_tokens":
            evaluator_output_tokens,

        "evaluator_total_tokens":
            (
                evaluator_input_tokens
                + evaluator_output_tokens
            ),
    }


# ============================================================
# COMPLETE RAG EVALUATION
# ============================================================

def evaluate_rag(
    question,
    generated_answer,
    retrieved_results,
):
    """
    Main RAGLABS evaluation function.

    Combines:

    Retrieval evaluation
        - Hit Rate
        - Reciprocal Rank

    Generation evaluation
        - Faithfulness
        - Correctness
        - Answer Relevance
    """

    golden_record = get_golden_record(
        question
    )

    # --------------------------------------------------------
    # If question belongs to controlled evaluation dataset
    # --------------------------------------------------------

    if golden_record:

        reference_answer = (
            golden_record[
                "reference_answer"
            ]
        )

        relevant_keywords = (
            golden_record[
                "relevant_keywords"
            ]
        )

        hit_rate = calculate_hit_rate(
            retrieved_results,
            relevant_keywords,
        )

        reciprocal_rank = (
            calculate_reciprocal_rank(
                retrieved_results,
                relevant_keywords,
            )
        )

    # --------------------------------------------------------
    # Unknown / user-created question
    # --------------------------------------------------------

    else:

        reference_answer = None
        relevant_keywords = None

        # Retrieval metrics require known relevance labels.
        hit_rate = None
        reciprocal_rank = None


    # --------------------------------------------------------
    # LLM Generation Evaluation
    # --------------------------------------------------------

    generation_metrics = (
        evaluate_generation(
            question=question,
            generated_answer=generated_answer,
            retrieved_results=retrieved_results,
            reference_answer=reference_answer,
        )
    )


    # --------------------------------------------------------
    # Final Evaluation Result
    # --------------------------------------------------------

    return {

        "hit_rate":
            hit_rate,

        "mrr":
            reciprocal_rank,

        "faithfulness":
            generation_metrics[
                "faithfulness"
            ],

        "correctness":
            generation_metrics[
                "correctness"
            ],

        "answer_relevance":
            generation_metrics[
                "answer_relevance"
            ],

        "reference_answer":
            reference_answer,

        "faithfulness_reason":
            generation_metrics[
                "faithfulness_reason"
            ],

        "correctness_reason":
            generation_metrics[
                "correctness_reason"
            ],

        "answer_relevance_reason":
            generation_metrics[
                "answer_relevance_reason"
            ],

        "evaluator_input_tokens":
            generation_metrics[
                "evaluator_input_tokens"
            ],

        "evaluator_output_tokens":
            generation_metrics[
                "evaluator_output_tokens"
            ],

        "evaluator_total_tokens":
            generation_metrics[
                "evaluator_total_tokens"
            ],

        "has_ground_truth":
            golden_record is not None,
    }