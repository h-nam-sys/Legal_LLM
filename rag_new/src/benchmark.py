"""
Benchmark retrieval.py using 50 queries:
- 25 on-topic
- 25 off-topic
- Hit@1 / Hit@3 / Hit@5
- MRR
- Latency Mean / P50 / P95 / Max
- Maximum raw Qdrant cosine score
- Threshold tuning 0.40 -> 0.85, step 0.05
- Precision / Recall / F1
- CSV results + threshold table + histogram

Important:
This benchmark follows the CURRENT retrieval.py pipeline:

    Query
      ↓
    detect_intent()
      ↓
    detect_procedure()
      ↓
    encode_query()
      ↓
    search_qdrant()
      ↓
    prepare_candidates()
      ↓
    rerank_candidates()
      ↓
    Final Top-K

Run from D:\\legal-rag:

    python src/benchmark.py
"""

import io
import time
import contextlib
from pathlib import Path
import re
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt

from retrieval import (
    load_model,
    connect_qdrant,
    detect_intent,
    detect_procedure,
    encode_query,
    search_qdrant,
    prepare_candidates,
    rerank_candidates,
    QDRANT_TOP_K,
    FINAL_TOP_K,
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "benchmark_50_queries.csv"
)

RESULT_DIR = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "results"
)


# =============================================================================
# THRESHOLDS
# =============================================================================

THRESHOLDS = [
    round(0.40 + 0.05 * i, 2)
    for i in range(10)
]


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text):
    """
    Normalize text only for evaluation matching.

    Does NOT modify the original query or Qdrant payload.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()

    # Vietnamese đ/Đ
    text = text.replace("đ", "d")

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# PROCEDURE MATCHING
# =============================================================================

def procedure_match(expected, actual):
    """
    Check whether the retrieved procedure matches the expected procedure.

    Exact normalized matching is intentionally used to avoid
    giving partial credit to an incorrect procedure.
    """

    if expected is None:
        return False

    expected = str(expected).strip()

    if expected.upper() == "NONE":
        return False

    if not actual:
        return False

    return normalize_text(expected) == normalize_text(actual)


# =============================================================================
# ONE QUERY
# =============================================================================

def run_one_query(
    query,
    model,
    client,
    procedure_names,
):
    """
    Run exactly the same logical retrieval pipeline as retrieval.py.

    Pipeline:

        detect_intent
            ↓
        detect_procedure
            ↓
        encode_query
            ↓
        search_qdrant
            ↓
        prepare_candidates
            ↓
        rerank_candidates

    expected_procedure is NOT passed into this function.
    Therefore benchmark does not leak the ground-truth answer
    into the retrieval pipeline.
    """

    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. Intent detection
    # -------------------------------------------------------------------------

    intent, intent_confidence = detect_intent(query)

    # -------------------------------------------------------------------------
    # 2. Procedure detection
    # -------------------------------------------------------------------------

    detected_procedure, procedure_confidence = detect_procedure(
        query,
        procedure_names,
    )

    # -------------------------------------------------------------------------
    # 3. Query embedding
    # -------------------------------------------------------------------------

    query_vector = encode_query(
        model,
        query,
    )

    # -------------------------------------------------------------------------
    # 4. Qdrant vector retrieval
    # -------------------------------------------------------------------------

    points = search_qdrant(
        client,
        query_vector,
        QDRANT_TOP_K,
    )

    # Keep RAW Qdrant cosine score.
    # Do NOT clamp this value because threshold tuning should use
    # the actual Qdrant similarity score.
    max_cosine_score = max(
        (
            float(getattr(point, "score", 0.0))
            for point in points
        ),
        default=0.0,
    )

    # -------------------------------------------------------------------------
    # 5. Candidate preparation
    # -------------------------------------------------------------------------

    # retrieval.py prints diagnostic information during normal retrieval.
    # Suppress it during benchmark execution so the benchmark output
    # remains readable.
    with contextlib.redirect_stdout(io.StringIO()):

        candidates = prepare_candidates(
            points,
            query,
            detected_procedure,
            intent,
        )

        # ---------------------------------------------------------------------
        # 6. Rerank ALL candidates
        # ---------------------------------------------------------------------

        final_results = rerank_candidates(
            candidates,
            FINAL_TOP_K,
        )

    # -------------------------------------------------------------------------
    # Latency
    # -------------------------------------------------------------------------

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "results": final_results,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "detected_procedure": detected_procedure,
        "procedure_confidence": procedure_confidence,
        "max_cosine_score": max_cosine_score,
        "latency_ms": latency_ms,
        "num_qdrant_candidates": len(points),
    }


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

def evaluate_hit_metrics(rows):
    """
    Calculate Hit@1, Hit@3, Hit@5 and MRR.

    Only ON-TOPIC queries are used for these retrieval metrics.

    Off-topic queries have no expected procedure, so they are not
    included in Hit@K / MRR.
    """

    on_rows = [
        row
        for row in rows
        if row["label"] == "on-topic"
    ]

    n = len(on_rows)

    if n == 0:
        return {
            "num_on_topic": 0,
            "hit_at_1_count": 0,
            "hit_at_1": 0.0,
            "hit_at_3_count": 0,
            "hit_at_3": 0.0,
            "hit_at_5_count": 0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
        }

    hit1 = sum(
        row["hit_at_1"]
        for row in on_rows
    )

    hit3 = sum(
        row["hit_at_3"]
        for row in on_rows
    )

    hit5 = sum(
        row["hit_at_5"]
        for row in on_rows
    )

    rr_sum = sum(
        row["reciprocal_rank"]
        for row in on_rows
    )

    return {
        "num_on_topic": n,

        "hit_at_1_count": hit1,
        "hit_at_1": hit1 / n,

        "hit_at_3_count": hit3,
        "hit_at_3": hit3 / n,

        "hit_at_5_count": hit5,
        "hit_at_5": hit5 / n,

        "mrr": rr_sum / n,
    }


# =============================================================================
# THRESHOLD EVALUATION
# =============================================================================

def evaluate_thresholds(rows):
    """
    Evaluate whether a query should be classified as on-topic
    based on maximum raw Qdrant cosine similarity.

    predicted_on_topic:
        max_cosine_score >= threshold
    """

    records = []

    for threshold in THRESHOLDS:

        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for row in rows:

            predicted_on_topic = (
                row["max_cosine_score"] >= threshold
            )

            actual_on_topic = (
                row["label"] == "on-topic"
            )

            if actual_on_topic and predicted_on_topic:
                tp += 1

            elif not actual_on_topic and predicted_on_topic:
                fp += 1

            elif actual_on_topic and not predicted_on_topic:
                fn += 1

            else:
                tn += 1

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        accuracy = (
            (tp + tn)
            / (tp + tn + fp + fn)
            if (tp + tn + fp + fn)
            else 0.0
        )

        records.append(
            {
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# LATENCY
# =============================================================================

def calculate_latency_summary(results_df):
    """
    Calculate latency statistics.
    """

    values = results_df["latency_ms"]

    return {
        "latency_mean_ms": values.mean(),
        "latency_p50_ms": values.quantile(0.50),
        "latency_p95_ms": values.quantile(0.95),
        "latency_max_ms": values.max(),
    }


# =============================================================================
# SCORE DISTRIBUTION PLOT
# =============================================================================

def save_score_distribution(rows):
    """
    Save histogram comparing maximum Qdrant cosine scores
    between on-topic and off-topic queries.
    """

    on_scores = [
        row["max_cosine_score"]
        for row in rows
        if row["label"] == "on-topic"
    ]

    off_scores = [
        row["max_cosine_score"]
        for row in rows
        if row["label"] == "off-topic"
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        on_scores,
        bins=10,
        alpha=0.6,
        label="On-topic",
    )

    plt.hist(
        off_scores,
        bins=10,
        alpha=0.6,
        label="Off-topic",
    )

    plt.xlabel(
        "Maximum Qdrant Cosine Similarity"
    )

    plt.ylabel(
        "Number of Queries"
    )

    plt.title(
        "Cosine Similarity Distribution: "
        "On-topic vs Off-topic"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULT_DIR
        / "cosine_score_distribution.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# =============================================================================
# THRESHOLD PLOT
# =============================================================================

def save_threshold_plot(threshold_df):
    """
    Save Precision / Recall / F1 against threshold.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1",
    )

    plt.xlabel(
        "Cosine Similarity Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Threshold Tuning"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULT_DIR
        / "threshold_tuning.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "VIETNAMESE LEGAL RAG - RETRIEVAL BENCHMARK"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Prepare result directory
    # -------------------------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load benchmark dataset
    # -------------------------------------------------------------------------

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark file not found:\n"
            f"{BENCHMARK_FILE}"
        )

    benchmark = pd.read_csv(
    BENCHMARK_FILE,
    encoding="utf-8-sig"
)

    required_columns = {
        "id",
        "query",
        "label",
        "expected_procedure",
    }

    missing_columns = (
        required_columns
        - set(benchmark.columns)
    )

    if missing_columns:

        raise ValueError(
            "Benchmark CSV is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # -------------------------------------------------------------------------
    # Dataset information
    # -------------------------------------------------------------------------

    total_queries = len(benchmark)

    on_topic_count = (
        benchmark["label"]
        .astype(str)
        .str.lower()
        .eq("on-topic")
        .sum()
    )

    off_topic_count = (
        benchmark["label"]
        .astype(str)
        .str.lower()
        .eq("off-topic")
        .sum()
    )

    print(
        f"\nBenchmark file : {BENCHMARK_FILE}"
    )

    print(
        f"Total queries  : {total_queries}"
    )

    print(
        f"On-topic       : {on_topic_count}"
    )

    print(
        f"Off-topic      : {off_topic_count}"
    )

    if total_queries != 50:

        print(
            "\n[WARNING] Expected 50 queries, "
            f"but found {total_queries}."
        )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------------------
    # Connect Qdrant
    # -------------------------------------------------------------------------

    client = connect_qdrant()

    # -------------------------------------------------------------------------
    # Load actual procedure names once
    # -------------------------------------------------------------------------

    # IMPORTANT:
    # This is exactly how retrieval.py is designed.
    #
    # We do NOT scan Qdrant for every query.
    # We load all unique procedure names once.

    from retrieval import load_procedure_names

    procedure_names = load_procedure_names(
        client
    )

    # -------------------------------------------------------------------------
    # Run benchmark
    # -------------------------------------------------------------------------

    rows = []

    try:

        for index, item in benchmark.iterrows():

            query_id = int(item["id"])

            query = str(
                item["query"]
            ).strip()

            label = str(
                item["label"]
            ).strip().lower()

            expected = str(
                item["expected_procedure"]
            ).strip()

            print()
            print(
                f"[{index + 1:02d}/{total_queries}] "
                f"{label.upper()} | {query}"
            )

            # -----------------------------------------------------------------
            # Run retrieval
            # -----------------------------------------------------------------

            output = run_one_query(
                query=query,
                model=model,
                client=client,
                procedure_names=procedure_names,
            )

            final_results = output["results"]

            intent = output["intent"]
            intent_confidence = (
                output["intent_confidence"]
            )

            detected_procedure = (
                output["detected_procedure"]
            )

            procedure_confidence = (
                output["procedure_confidence"]
            )

            max_cosine_score = (
                output["max_cosine_score"]
            )

            latency_ms = (
                output["latency_ms"]
            )

            num_qdrant_candidates = (
                output["num_qdrant_candidates"]
            )

            # -----------------------------------------------------------------
            # Evaluate ranking
            # -----------------------------------------------------------------

            ranks = []

            top_procedure = ""

            top_final_score = None

            if final_results:

                top_procedure = (
                    final_results[0]["data"]
                    .get("procedure", "")
                )

                top_final_score = (
                    final_results[0]
                    .get("final_score")
                )

            for rank, result_item in enumerate(
                final_results,
                start=1,
            ):

                data = result_item["data"]

                actual_procedure = data.get(
                    "procedure",
                    "",
                )

                if procedure_match(
                    expected,
                    actual_procedure,
                ):

                    ranks.append(rank)

            first_correct_rank = (
                min(ranks)
                if ranks
                else None
            )

            hit_at_1 = int(
                first_correct_rank is not None
                and first_correct_rank <= 1
            )

            hit_at_3 = int(
                first_correct_rank is not None
                and first_correct_rank <= 3
            )

            hit_at_5 = int(
                first_correct_rank is not None
                and first_correct_rank <= 5
            )

            reciprocal_rank = (
                1.0 / first_correct_rank
                if first_correct_rank is not None
                else 0.0
            )

            # -----------------------------------------------------------------
            # Save row
            # -----------------------------------------------------------------

            rows.append(
                {
                    "id": query_id,
                    "query": query,
                    "label": label,
                    "expected_procedure": expected,

                    "detected_intent": intent,
                    "intent_confidence": (
                        intent_confidence
                    ),

                    "detected_procedure": (
                        detected_procedure
                        if detected_procedure
                        else ""
                    ),

                    "procedure_confidence": (
                        procedure_confidence
                    ),

                    "top1_procedure": (
                        top_procedure
                    ),

                    "top1_final_score": (
                        top_final_score
                        if top_final_score is not None
                        else 0.0
                    ),

                    "first_correct_rank": (
                        first_correct_rank
                        if first_correct_rank is not None
                        else ""
                    ),

                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "hit_at_5": hit_at_5,

                    "reciprocal_rank": (
                        reciprocal_rank
                    ),

                    "max_cosine_score": (
                        max_cosine_score
                    ),

                    "num_qdrant_candidates": (
                        num_qdrant_candidates
                    ),

                    "latency_ms": (
                        latency_ms
                    ),
                }
            )

            # -----------------------------------------------------------------
            # Console output
            # -----------------------------------------------------------------

            print(
                f"    Intent        : "
                f"{intent} "
                f"({intent_confidence:.4f})"
            )

            print(
                f"    Procedure     : "
                f"{detected_procedure}"
            )

            print(
                f"    max cosine   : "
                f"{max_cosine_score:.4f}"
            )

            print(
                f"    latency      : "
                f"{latency_ms:.2f} ms"
            )

            if first_correct_rank is not None:

                print(
                    f"    correct rank : "
                    f"{first_correct_rank}"
                )

            elif label == "on-topic":

                print(
                    "    correct rank : "
                    "NOT FOUND"
                )

            else:

                print(
                    "    off-topic    : "
                    "no expected procedure"
                )

    finally:

        client.close()

    # =============================================================================
    # DATAFRAME
    # =============================================================================

    results_df = pd.DataFrame(
        rows
    )

    # =============================================================================
    # RETRIEVAL METRICS
    # =============================================================================

    hit_metrics = evaluate_hit_metrics(
        rows
    )

    # =============================================================================
    # LATENCY
    # =============================================================================

    latency_summary = (
        calculate_latency_summary(
            results_df
        )
    )

    # =============================================================================
    # THRESHOLD TUNING
    # =============================================================================

    threshold_df = evaluate_thresholds(
        rows
    )

    # Find best threshold by F1.
    best_idx = threshold_df[
        "f1"
    ].idxmax()

    best = threshold_df.loc[
        best_idx
    ]

    # =============================================================================
    # SAVE CSV
    # =============================================================================

    results_path = (
        RESULT_DIR
        / "benchmark_results.csv"
    )

    threshold_path = (
        RESULT_DIR
        / "threshold_results.csv"
    )

    summary_path = (
        RESULT_DIR
        / "benchmark_summary.csv"
    )

    # Detailed query-level results
    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Threshold results
    threshold_df.to_csv(
        threshold_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =============================================================================
    # SUMMARY CSV
    # =============================================================================

    summary_rows = [

        {
            "metric": "Total Queries",
            "value": total_queries,
        },

        {
            "metric": "On-topic Queries",
            "value": on_topic_count,
        },

        {
            "metric": "Off-topic Queries",
            "value": off_topic_count,
        },

        {
            "metric": "Hit@1",
            "value": hit_metrics["hit_at_1"],
        },

        {
            "metric": "Hit@3",
            "value": hit_metrics["hit_at_3"],
        },

        {
            "metric": "Hit@5",
            "value": hit_metrics["hit_at_5"],
        },

        {
            "metric": "MRR",
            "value": hit_metrics["mrr"],
        },

        {
            "metric": "Latency Mean (ms)",
            "value": latency_summary[
                "latency_mean_ms"
            ],
        },

        {
            "metric": "Latency P50 (ms)",
            "value": latency_summary[
                "latency_p50_ms"
            ],
        },

        {
            "metric": "Latency P95 (ms)",
            "value": latency_summary[
                "latency_p95_ms"
            ],
        },

        {
            "metric": "Latency Max (ms)",
            "value": latency_summary[
                "latency_max_ms"
            ],
        },

        {
            "metric": "Best Threshold",
            "value": best[
                "threshold"
            ],
        },

        {
            "metric": "Best Precision",
            "value": best[
                "precision"
            ],
        },

        {
            "metric": "Best Recall",
            "value": best[
                "recall"
            ],
        },

        {
            "metric": "Best F1",
            "value": best[
                "f1"
            ],
        },

        {
            "metric": "Best Accuracy",
            "value": best[
                "accuracy"
            ],
        },
    ]

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # =============================================================================
    # SAVE PLOTS
    # =============================================================================

    score_plot = (
        save_score_distribution(
            rows
        )
    )

    threshold_plot = (
        save_threshold_plot(
            threshold_df
        )
    )

    # =============================================================================
    # CONSOLE SUMMARY
    # =============================================================================

    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Queries       : "
        f"{total_queries}"
    )

    print(
        f"On-topic      : "
        f"{on_topic_count}"
    )

    print(
        f"Off-topic     : "
        f"{off_topic_count}"
    )

    print()

    print(
        f"Hit@1 : "
        f"{hit_metrics['hit_at_1_count']}/"
        f"{hit_metrics['num_on_topic']} "
        f"({hit_metrics['hit_at_1']:.2%})"
    )

    print(
        f"Hit@3 : "
        f"{hit_metrics['hit_at_3_count']}/"
        f"{hit_metrics['num_on_topic']} "
        f"({hit_metrics['hit_at_3']:.2%})"
    )

    print(
        f"Hit@5 : "
        f"{hit_metrics['hit_at_5_count']}/"
        f"{hit_metrics['num_on_topic']} "
        f"({hit_metrics['hit_at_5']:.2%})"
    )

    print(
        f"MRR   : "
        f"{hit_metrics['mrr']:.4f}"
    )

    print()

    print(
        f"Latency Mean : "
        f"{latency_summary['latency_mean_ms']:.2f} ms"
    )

    print(
        f"Latency P50  : "
        f"{latency_summary['latency_p50_ms']:.2f} ms"
    )

    print(
        f"Latency P95  : "
        f"{latency_summary['latency_p95_ms']:.2f} ms"
    )

    print(
        f"Latency Max  : "
        f"{latency_summary['latency_max_ms']:.2f} ms"
    )

    # =============================================================================
    # THRESHOLD TABLE
    # =============================================================================

    print()
    print("-" * 80)
    print("THRESHOLD TUNING")
    print("-" * 80)

    display_threshold_df = threshold_df[
        [
            "threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]
    ].copy()

    print(
        display_threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =============================================================================
    # BEST THRESHOLD
    # =============================================================================

    print()
    print(
        f"RECOMMENDED THRESHOLD: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best['f1']:.4f}"
    )

    print(
        f"Accuracy  : "
        f"{best['accuracy']:.4f}"
    )

    # =============================================================================
    # OUTPUT FILES
    # =============================================================================

    print()
    print("-" * 80)
    print("FILES GENERATED")
    print("-" * 80)

    print(
        f"- {results_path}"
    )

    print(
        f"- {threshold_path}"
    )

    print(
        f"- {summary_path}"
    )

    print(
        f"- {score_plot}"
    )

    print(
        f"- {threshold_plot}"
    )

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[INFO] Benchmark interrupted by user."
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print("[FATAL ERROR]")
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print("=" * 80)

        raise