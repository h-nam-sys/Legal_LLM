from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_chunks.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "administrative_procedures_embeddings.parquet"
)

MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

BATCH_SIZE = 8


# ============================================================
# HEADER
# ============================================================

def print_header():
    print("=" * 80)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("BKAI VIETNAMESE BI-ENCODER EMBEDDING")
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks():
    print("\nLoading chunks...")
    print(f"Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(f"Documents/chunks loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    if "text" not in df.columns:
        raise ValueError("Column 'text' not found.")

    df["chunk_text"] = df["text"].fillna("").astype(str)

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):
    print("\n" + "=" * 80)
    print("INPUT VALIDATION")
    print("=" * 80)

    missing = df["chunk_text"].isna().sum()

    empty = (
        df["chunk_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nRows              : {len(df):,}")
    print(f"Missing chunk_text: {missing:,}")
    print(f"Empty chunk_text  : {empty:,}")

    if missing > 0:
        raise ValueError("Some chunks have missing text.")

    if empty > 0:
        raise ValueError("Some chunks have empty text.")

    print("\n[OK] Input validation passed")


# ============================================================
# VIETNAMESE WORD SEGMENTATION
# ============================================================

def segment_text(text):
    """
    BKAI Vietnamese Bi-Encoder requires
    Vietnamese text to be word-segmented.
    """

    text = str(text).strip()

    if not text:
        return ""

    return ViTokenizer.tokenize(text)


def prepare_texts(df):
    print("\n" + "=" * 80)
    print("VIETNAMESE WORD SEGMENTATION")
    print("=" * 80)

    print("\nSegmenting chunk texts...")

    segmented_texts = []

    for text in tqdm(
        df["chunk_text"].tolist(),
        desc="Word segmentation"
    ):
        segmented_texts.append(
            segment_text(text)
        )

    df = df.copy()

    df["segmented_text"] = segmented_texts

    print("\n[OK] Word segmentation completed")

    return df


# ============================================================
# LOAD BKAI MODEL
# ============================================================

def load_model():
    print("\n" + "=" * 80)
    print("LOADING BKAI MODEL")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")

    if torch.cuda.is_available():
        device = "cuda"

        print("CUDA available: True")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"

        print("CUDA available: False")
        print("Using CPU")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    dimension = model.get_sentence_embedding_dimension()

    print(f"Embedding dimension: {dimension}")
    print(f"Device: {device}")

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(df, model):
    print("\n" + "=" * 80)
    print("CREATING BKAI EMBEDDINGS")
    print("=" * 80)

    texts = (
        df["segmented_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"\nNumber of chunks: {len(texts):,}")
    print(f"Batch size       : {BATCH_SIZE}")

    print("\nEncoding...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("\nEmbedding shape:")
    print(embeddings.shape)

    print(f"Embedding dtype: {embeddings.dtype}")

    return embeddings


# ============================================================
# SAVE DATASET
# ============================================================

def save_embeddings(df, embeddings):
    print("\n" + "=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    output_df = df.copy()

    output_df["embedding"] = [
        vector.tolist()
        for vector in embeddings
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow"
    )

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print(f"\nRows: {len(output_df):,}")

    print(
        "Embedding dimension: "
        f"{len(output_df['embedding'].iloc[0])}"
    )


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(embeddings):
    print("\n" + "=" * 80)
    print("EMBEDDING VALIDATION")
    print("=" * 80)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(f"\nNumber of vectors : {len(embeddings):,}")
    print(f"Vector dimension  : {embeddings.shape[1]:,}")

    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()

    print(f"Contains NaN      : {has_nan}")
    print(f"Contains Inf      : {has_inf}")

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    print("\nVector norm statistics:")
    print(f"Min    : {norms.min():.6f}")
    print(f"Max    : {norms.max():.6f}")
    print(f"Mean   : {norms.mean():.6f}")

    print("\nQUALITY CHECK")

    if embeddings.shape[1] == 768:
        print("[OK] Embedding dimension = 768")
    else:
        print(
            "[WARNING] Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    if not has_nan:
        print("[OK] No NaN values")

    if not has_inf:
        print("[OK] No Inf values")

    if np.allclose(
        norms,
        1.0,
        atol=1e-4
    ):
        print("[OK] Embeddings are normalized")
    else:
        print("[WARNING] Embeddings are not normalized")


# ============================================================
# SAMPLE
# ============================================================

def show_sample(df):
    print("\n" + "=" * 80)
    print("SAMPLE EMBEDDING")
    print("=" * 80)

    row = df.iloc[0]

    print("\nChunk ID:")
    print(row["chunk_id"])

    print("\nProcedure:")
    print(row["procedure_name"])

    print("\nChunk type:")
    print(row["chunk_type"])

    print("\nOriginal text:")
    print(row["chunk_text"])

    print("\nSegmented text:")
    print(row["segmented_text"])

    print("\nEmbedding dimension:")
    print(len(row["embedding"]))

    print("\nFirst 10 embedding values:")
    print(row["embedding"][:10])


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # 1. Load chunks
    df = load_chunks()

    # 2. Validate
    validate_input(df)

    # 3. Vietnamese word segmentation
    df = prepare_texts(df)

    # 4. Load BKAI
    model = load_model()

    # 5. Create embeddings
    embeddings = create_embeddings(
        df,
        model
    )

    # 6. Validate embeddings
    validate_embeddings(
        embeddings
    )

    # 7. Save
    save_embeddings(
        df,
        embeddings
    )

    # 8. Add embeddings for sample display
    df["embedding"] = [
        vector.tolist()
        for vector in embeddings
    ]

    # 9. Show sample
    show_sample(df)

    print("\n" + "=" * 80)
    print("EMBEDDING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()