from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
)

OUTPUT_NPY = OUTPUT_DIR / "embeddings.npy"
OUTPUT_JSONL = OUTPUT_DIR / "embeddings.jsonl"
OUTPUT_METADATA = OUTPUT_DIR / "embedding_metadata.json"


# -------------------------------------------------------------------------
# Embedding model
# -------------------------------------------------------------------------
#
# Multilingual model, suitable for Vietnamese text.
#
# We keep the model configurable through this variable so that later
# we can benchmark another Vietnamese / multilingual embedding model
# without changing the pipeline.
#
MODEL_NAME = "intfloat/multilingual-e5-base"


# E5 models expect different prefixes for query and document.
#
# This file embeds documents/chunks, therefore we use:
#
#     passage: ...
#
DOCUMENT_PREFIX = "passage: "


# -------------------------------------------------------------------------
# Embedding settings
# -------------------------------------------------------------------------

BATCH_SIZE = 16

NORMALIZE_EMBEDDINGS = True

SHOW_PROGRESS = True


# =============================================================================
# DEVICE
# =============================================================================

def detect_device() -> str:
    """
    Automatically select CUDA when available.

    Returns:
        "cuda" if NVIDIA CUDA is available.
        "cpu" otherwise.
    """

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# =============================================================================
# ENVIRONMENT INFO
# =============================================================================

def print_environment(device: str) -> None:

    print()
    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device        : {device}")

    if torch.cuda.is_available():

        print(
            f"CUDA version  : "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU count     : "
            f"{torch.cuda.device_count()}"
        )


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_input_dataframe(
    df: pd.DataFrame,
) -> None:

    required_columns = {
        "chunk_id",
        "procedure_id",
        "procedure_name",
        "domain",
        "intent",
        "chunk_type",
        "text",
        "search_text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing_columns)
            )
        )

    if df.empty:

        raise ValueError(
            "Input chunks.csv is empty."
        )

    if df["chunk_id"].duplicated().any():

        duplicated = (
            df.loc[
                df["chunk_id"].duplicated(keep=False),
                "chunk_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate chunk_id detected:\n"
            + "\n".join(
                f"  - {chunk_id}"
                for chunk_id in duplicated
            )
        )

    for column in [
        "procedure_id",
        "procedure_name",
        "intent",
        "text",
    ]:

        empty_mask = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        )

        if empty_mask.any():

            rows = (
                df.index[empty_mask]
                .tolist()
            )

            raise ValueError(
                f"Empty values detected in "
                f"column '{column}' "
                f"at rows {rows[:20]}"
            )


# =============================================================================
# EMBEDDING TEXT
# =============================================================================

def build_embedding_text(
    row: pd.Series,
) -> str:
    """
    Build the exact document text sent to the embedding model.

    We intentionally use the structured chunk text generated
    by Phase 2.

    We do NOT summarize or rewrite legal content here.
    """

    text = str(
        row["text"]
    ).strip()

    return (
        DOCUMENT_PREFIX
        + text
    )


# =============================================================================
# MODEL
# =============================================================================

def load_model(
    model_name: str,
    device: str,
) -> SentenceTransformer:

    print()
    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    print(f"Model : {model_name}")
    print(f"Device: {device}")

    start_time = time.time()

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    elapsed = time.time() - start_time

    print(
        f"Model loaded in "
        f"{elapsed:.2f} seconds"
    )

    # Display embedding dimension
    dimension = model.get_sentence_embedding_dimension()

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    return model


# =============================================================================
# EMBEDDING
# =============================================================================

def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:

    print()
    print("=" * 80)
    print("GENERATING EMBEDDINGS")
    print("=" * 80)

    print(
        f"Texts      : {len(texts)}"
    )

    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        f"Normalize  : {NORMALIZE_EMBEDDINGS}"
    )

    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=SHOW_PROGRESS,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start_time

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print()
    print(
        f"Embedding completed in "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"Dtype: {embeddings.dtype}"
    )

    return embeddings


# =============================================================================
# EMBEDDING VALIDATION
# =============================================================================

def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: int,
) -> None:

    print()
    print("=" * 80)
    print("EMBEDDING VALIDATION")
    print("=" * 80)

    if embeddings.ndim != 2:

        raise ValueError(
            f"Embeddings must be 2D, "
            f"got shape {embeddings.shape}"
        )

    rows, dimension = embeddings.shape

    print(
        f"Rows      : {rows}"
    )

    print(
        f"Dimension : {dimension}"
    )

    print(
        f"Dtype     : {embeddings.dtype}"
    )

    # Number of vectors must match chunks
    if rows != expected_rows:

        raise ValueError(
            f"Embedding count mismatch: "
            f"{rows} vectors for "
            f"{expected_rows} chunks."
        )

    # NaN
    if np.isnan(embeddings).any():

        raise ValueError(
            "Embeddings contain NaN values."
        )

    # Infinity
    if np.isinf(embeddings).any():

        raise ValueError(
            "Embeddings contain infinite values."
        )

    # Zero vectors
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    zero_vectors = np.sum(
        norms == 0
    )

    if zero_vectors > 0:

        raise ValueError(
            f"Found {zero_vectors} "
            f"zero vectors."
        )

    print(
        "NaN values     : 0"
    )

    print(
        "Infinite values: 0"
    )

    print(
        f"Zero vectors   : {zero_vectors}"
    )

    print("PASS")


# =============================================================================
# SAVE
# =============================================================================

def save_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Save numpy matrix
    # -------------------------------------------------------------------------

    np.save(
        OUTPUT_NPY,
        embeddings,
    )

    # -------------------------------------------------------------------------
    # 2. Save metadata + vectors as JSONL
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8",
    ) as f:

        for index, row in df.iterrows():

            record = {
                "index": int(index),
                "chunk_id": str(
                    row["chunk_id"]
                ),
                "procedure_id": str(
                    row["procedure_id"]
                ),
                "procedure_name": str(
                    row["procedure_name"]
                ),
                "domain": str(
                    row["domain"]
                ),
                "intent": str(
                    row["intent"]
                ),
                "chunk_type": str(
                    row["chunk_type"]
                ),
                "text": str(
                    row["text"]
                ),
                "embedding": (
                    embeddings[index]
                    .tolist()
                ),
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -------------------------------------------------------------------------
    # 3. Save metadata
    # -------------------------------------------------------------------------

    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "vector_count": int(
            embeddings.shape[0]
        ),
        "normalize_embeddings": (
            NORMALIZE_EMBEDDINGS
        ),
        "document_prefix": DOCUMENT_PREFIX,
        "source_chunks": str(
            INPUT_FILE
        ),
        "output_npy": str(
            OUTPUT_NPY
        ),
        "output_jsonl": str(
            OUTPUT_JSONL
        ),
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"NPY      : {OUTPUT_NPY}"
    )

    print(
        f"JSONL    : {OUTPUT_JSONL}"
    )

    print(
        f"Metadata : {OUTPUT_METADATA}"
    )


# =============================================================================
# SAMPLE
# =============================================================================

def print_samples(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:

    print()
    print("=" * 80)
    print("EMBEDDING SAMPLE")
    print("=" * 80)

    sample_indices = [
        0,
        min(1, len(df) - 1),
        min(2, len(df) - 1),
    ]

    already_seen = set()

    for index in sample_indices:

        if index in already_seen:
            continue

        already_seen.add(index)

        row = df.iloc[index]

        vector = embeddings[index]

        print()
        print("-" * 80)

        print(
            f"Index     : {index}"
        )

        print(
            f"Chunk ID   : {row['chunk_id']}"
        )

        print(
            f"Procedure : {row['procedure_name']}"
        )

        print(
            f"Intent    : {row['intent']}"
        )

        print(
            f"Dimension : {len(vector)}"
        )

        print(
            "Vector[:10]:"
        )

        print(
            vector[:10]
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LEGAL RAG - EMBEDDING / PHASE 3")
    print("=" * 80)

    print(
        f"Input: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = detect_device()

    print_environment(device)

    # -------------------------------------------------------------------------
    # Load chunks
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING CHUNKS")
    print("=" * 80)

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    print(
        f"Loaded {len(df)} chunks."
    )

    validate_input_dataframe(df)

    # -------------------------------------------------------------------------
    # Build embedding texts
    # -------------------------------------------------------------------------

    texts = [
        build_embedding_text(row)
        for _, row in df.iterrows()
    ]

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = load_model(
        model_name=MODEL_NAME,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Generate embeddings
    # -------------------------------------------------------------------------

    embeddings = generate_embeddings(
        model=model,
        texts=texts,
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_embeddings(
        embeddings=embeddings,
        expected_rows=len(df),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_embeddings(
        df=df,
        embeddings=embeddings,
        model_name=MODEL_NAME,
    )

    # -------------------------------------------------------------------------
    # Samples
    # -------------------------------------------------------------------------

    print_samples(
        df=df,
        embeddings=embeddings,
    )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 3 COMPLETED")
    print("=" * 80)

    print(
        f"Vectors generated: "
        f"{len(embeddings)}"
    )

    print(
        f"Dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Device: {device}"
    )


if __name__ == "__main__":
    main()