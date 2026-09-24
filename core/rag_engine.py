import os
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI


load_dotenv()


def process_pdfs(uploaded_files, chunk_size=500, chunk_overlap=100):
    """
    Load uploaded PDFs and split them into chunks.
    """

    documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            # Add source filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(docs)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def build_retrievers(
    chunks,
    bm25_k=5,
    vector_k=5,
):
    """
    Build BM25 and FAISS vector retrievers.
    """

    # -------------------------
    # BM25 Retriever
    # -------------------------

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k

    # -------------------------
    # Vector Retriever
    # -------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": vector_k}
    )

    return bm25_retriever, vector_retriever


def hybrid_retrieve(
    query,
    bm25_retriever,
    vector_retriever,
    bm25_weight=0.4,
    vector_weight=0.6,
    final_k=5,
):
    """
    Combine BM25 + Vector retrieval using
    weighted Reciprocal Rank Fusion (RRF).
    """

    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    scores = {}
    documents = {}

    # RRF constant
    rrf_k = 60

    # -------------------------
    # BM25 ranking
    # -------------------------

    for rank, doc in enumerate(bm25_docs, start=1):

        key = (
            doc.page_content,
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )

        score = bm25_weight * (
            1 / (rrf_k + rank)
        )

        scores[key] = scores.get(key, 0) + score
        documents[key] = doc

    # -------------------------
    # Vector ranking
    # -------------------------

    for rank, doc in enumerate(vector_docs, start=1):

        key = (
            doc.page_content,
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )

        score = vector_weight * (
            1 / (rrf_k + rank)
        )

        scores[key] = scores.get(key, 0) + score
        documents[key] = doc

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    results = []

    for key in ranked_keys[:final_k]:

        doc = documents[key]

        results.append(
            {
                "document": doc,
                "fusion_score": scores[key],
            }
        )

    return results


def generate_answer(query, retrieved_results):
    """
    Generate an answer using retrieved context.
    """

    context = "\n\n".join(
        result["document"].page_content
        for result in retrieved_results
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = f"""
You are the answer-generation component of RAGLABS.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "The provided knowledge base does not contain this information."
4. Keep the answer concise and factual.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    # Token usage
    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens

    return {
        "answer": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }