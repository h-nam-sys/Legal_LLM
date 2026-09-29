# -*- coding: utf-8 -*-
"""
VIETNAMESE LEGAL RAG - RETRIEVAL BENCHMARK

Benchmark source:
    data/benchmark/legal_rag_benchmark_84_plus_robustness.xlsx

Supported sheets:
    - Benchmark_84
    - Robustness_Queries

Metrics:
    - Hit@1
    - Hit@3
    - Hit@5
    - MRR
    - Procedure detection accuracy
    - Latency Mean / P50 / P95 / Max
    - Category summary

The benchmark calls the real retrieval pipeline through:
    retrieval.retrieve_with_diagnostics()

Ground truth is NEVER passed into retrieval.
"""

from __future__ import annotations

import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
# Force a non-interactive backend before pyplot is imported: this script
# only ever calls savefig()/close(), never show(), and running headless
# (no display attached, e.g. over SSH/CI) would otherwise raise on some
# default backends.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import retrieval


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "legal_rag_benchmark_84_plus_robustness.xlsx"
)

RESULT_DIR = (
    BASE_DIR
    / "data"
    / "benchmark"
    / "results"
)

DETAILED_CSV = (
    RESULT_DIR
    / "benchmark_detailed_results.csv"
)

CATEGORY_CSV = (
    RESULT_DIR
    / "benchmark_category_summary.csv"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "benchmark_summary.csv"
)

REPORT_XLSX = (
    RESULT_DIR
    / "benchmark_report.xlsx"
)

LATENCY_PNG = (
    RESULT_DIR
    / "latency_distribution.png"
)

FINAL_SCORE_PNG = (
    RESULT_DIR
    / "final_score_distribution.png"
)


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(text: Any) -> str:
    """
    Normalize only for benchmark evaluation.
    """

    if text is None:
        return ""

    value = str(text).strip().lower()

    value = value.replace(
        "đ",
        "d",
    )

    value = unicodedata.normalize(
        "NFD",
        value,
    )

    value = "".join(
        char
        for char in value
        if unicodedata.category(char) != "Mn"
    )

    value = " ".join(
        value.split()
    )

    return value


def is_none_expected(value: Any) -> bool:
    if value is None:
        return True

    text = normalize_text(value)

    return text in {
        "",
        "none",
        "null",
        "nan",
        "n/a",
        "na",
    }


def procedure_match(
    expected: Any,
    actual: Optional[str],
) -> bool:

    if is_none_expected(expected):
        return (
            actual is None
            or not str(actual).strip()
        )

    if not actual:
        return False

    return (
        normalize_text(expected)
        == normalize_text(actual)
    )


# =============================================================================
# DATASET COLUMN HELPERS
# =============================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    normalized_columns = {
        normalize_text(column): column
        for column in df.columns
    }

    for candidate in candidates:

        key = normalize_text(
            candidate
        )

        if key in normalized_columns:
            return normalized_columns[key]

    return None


def prepare_sheet(
    df: pd.DataFrame,
    sheet_name: str,
) -> pd.DataFrame:

    if df.empty:
        return df

    query_col = find_column(
        df,
        [
            "query",
            "question",
            "user_query",
            "câu hỏi",
            "cau hoi",
        ],
    )

    if query_col is None:
        raise ValueError(
            f"Sheet '{sheet_name}' does not contain "
            "a query column."
        )

    expected_col = find_column(
        df,
        [
            "expected_procedure",
            "expected procedure",
            "expected",
            "procedure",
            "gold_procedure",
            "gold procedure",
            "thủ tục mong đợi",
        ],
    )

    id_col = find_column(
        df,
        [
            "id",
            "query_id",
            "case_id",
        ],
    )

    category_col = find_column(
        df,
        [
            "category",
            "type",
            "group",
            "nhóm",
        ],
    )

    label_col = find_column(
        df,
        [
            "label",
            "scope",
        ],
    )

    result = pd.DataFrame()

    result["query"] = (
        df[query_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if id_col:
        result["id"] = df[id_col]
    else:
        result["id"] = range(
            1,
            len(df) + 1,
        )

    if expected_col:
        result["expected_procedure"] = (
            df[expected_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        result["expected_procedure"] = ""

    if category_col:
        result["category"] = (
            df[category_col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )
    else:
        result["category"] = "unknown"

    if label_col:
        result["label"] = (
            df[label_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        result["label"] = ""

    result["sheet"] = sheet_name

    result = result[
        result["query"].str.len() > 0
    ].copy()

    return result


def load_benchmark() -> pd.DataFrame:

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            f"Benchmark file not found:\n"
            f"{BENCHMARK_FILE}"
        )

    sheets = pd.read_excel(
        BENCHMARK_FILE,
        sheet_name=None,
    )

    all_frames = []

    for sheet_name, df in sheets.items():

        # README / documentation sheets do not contain benchmark queries.
        # Skip them instead of treating them as benchmark data.
        query_col = find_column(
            df,
            [
                "query",
                "question",
                "user_query",
                "câu hỏi",
                "cau hoi",
            ],
        )

        if query_col is None:
            print(
                f"[INFO] Skipping sheet '{sheet_name}' "
                "(no query column)."
            )
            continue

        prepared = prepare_sheet(
            df,
            sheet_name,
        )

        if not prepared.empty:
            all_frames.append(
                prepared
            )

    if not all_frames:
        raise ValueError(
            "No usable benchmark queries "
            "were found in the workbook."
        )

    benchmark = pd.concat(
        all_frames,
        ignore_index=True,
    )

    return benchmark


# =============================================================================
# QDRANT CHECK
# =============================================================================

def get_point_count(
    client: Any,
) -> int:

    collection_info = client.get_collection(
        retrieval.COLLECTION_NAME
    )

    points_count = getattr(
        collection_info,
        "points_count",
        None,
    )

    if points_count is None:
        points_count = getattr(
            collection_info,
            "vectors_count",
            None,
        )

    if points_count is None:
        return 0

    return int(points_count)


# =============================================================================
# HIT@K / MRR
# =============================================================================

def result_procedures(
    results: List[Any],
) -> List[str]:

    return [
        str(result.procedure).strip()
        for result in results
        if str(result.procedure).strip()
    ]


def correct_rank(
    expected: Any,
    results: List[Any],
) -> Optional[int]:

    if is_none_expected(expected):
        return None

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if procedure_match(
            expected,
            result.procedure,
        ):
            return rank

    return None


def hit_at_k(
    expected: Any,
    results: List[Any],
    k: int,
) -> int:

    if is_none_expected(expected):
        return 0

    for result in results[:k]:
        if procedure_match(
            expected,
            result.procedure,
        ):
            return 1

    return 0


def reciprocal_rank(
    expected: Any,
    rank: Optional[int],
) -> float:

    if is_none_expected(expected):
        return 0.0

    if rank is None:
        return 0.0

    return 1.0 / rank


# =============================================================================
# LATENCY
# =============================================================================

def percentile(
    values: List[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    position = (
        (len(values) - 1)
        * p
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = (
        position - lower
    )

    return (
        values[lower]
        * (1 - weight)
        + values[upper]
        * weight
    )


# =============================================================================
# CATEGORY HELPERS
# =============================================================================

def is_out_of_domain(
    row: pd.Series,
) -> bool:

    category = normalize_text(
        row.get("category", "")
    )

    label = normalize_text(
        row.get("label", "")
    )

    return (
        "out_of_domain" in category
        or "out-of-domain" in category
        or "out of domain" in category
        or "off_topic" in category
        or "off-topic" in category
        or "off topic" in category
        or "out_of_domain" in label
        or "off-topic" in label
    )


def is_ambiguous_case(
    row: pd.Series,
) -> bool:

    category = normalize_text(
        row.get("category", "")
    )

    return (
        "ambiguous" in category
        or "ambigu" in normalize_text(
            row.get("label", "")
        )
    )


# =============================================================================
# RUN ONE QUERY
# =============================================================================

def run_one_query(
    query: str,
    model: Any,
    client: Any,
    procedures: List[str],
) -> Dict[str, Any]:

    start = time.perf_counter()

    output = retrieval.retrieve_with_diagnostics(
        query=query,
        model=model,
        client=client,
        procedures=procedures,
        top_k=retrieval.QDRANT_TOP_K,
        final_top_k=retrieval.FINAL_TOP_K,
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return {
        **output,
        "latency_ms": latency_ms,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "VIETNAMESE LEGAL RAG - RETRIEVAL BENCHMARK"
    )
    print("=" * 100)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark = load_benchmark()

    print()
    print(
        f"Benchmark workbook : {BENCHMARK_FILE}"
    )
    print(
        f"Total queries      : {len(benchmark)}"
    )
    print(
        f"Sheets             : "
        f"{', '.join(sorted(benchmark['sheet'].unique()))}"
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = retrieval.load_model()

    # -------------------------------------------------------------------------
    # Connect Qdrant Embedded
    # -------------------------------------------------------------------------

    client = retrieval.connect_qdrant()

    latencies: List[float] = []
    final_scores: List[float] = []

    rows: List[Dict[str, Any]] = []

    try:

        point_count = get_point_count(
            client
        )

        print()
        print(
            f"Qdrant collection: "
            f"{retrieval.COLLECTION_NAME}"
        )
        print(
            f"Qdrant points    : {point_count}"
        )

        if point_count <= 0:
            raise RuntimeError(
                "Qdrant collection contains 0 points. "
                "Benchmark stopped."
            )

        # -------------------------------------------------------------
        # Load procedure index ONCE.
        # -------------------------------------------------------------

        procedures = (
            retrieval.load_procedure_names(
                client
            )
        )

        if not procedures:
            raise RuntimeError(
                "No procedure names found in Qdrant."
            )

        print()
        print(
            f"Procedure index size: "
            f"{len(procedures)}"
        )

        # -------------------------------------------------------------
        # Benchmark loop
        # -------------------------------------------------------------

        total_queries = len(
            benchmark
        )

        for index, row in benchmark.iterrows():

            query_id = row["id"]

            query = str(
                row["query"]
            ).strip()

            expected = str(
                row.get(
                    "expected_procedure",
                    "",
                )
            ).strip()

            category = str(
                row.get(
                    "category",
                    "unknown",
                )
            ).strip()

            label = str(
                row.get(
                    "label",
                    "",
                )
            ).strip()

            print()
            print("-" * 100)
            print(
                f"[{index + 1}/{total_queries}] "
                f"{query}"
            )

            output = run_one_query(
                query=query,
                model=model,
                client=client,
                procedures=procedures,
            )

            results = output["results"]

            detected_intent = (
                output[
                    "detected_intent"
                ]
            )

            intent_confidence = (
                output[
                    "intent_confidence"
                ]
            )

            detected_procedure = (
                output[
                    "detected_procedure"
                ]
            )

            procedure_confidence = (
                output[
                    "procedure_confidence"
                ]
            )

            latency_ms = output[
                "latency_ms"
            ]

            latencies.append(
                latency_ms
            )

            # ---------------------------------------------------------
            # Retrieval ranking
            # ---------------------------------------------------------

            rank = correct_rank(
                expected,
                results,
            )

            hit1 = hit_at_k(
                expected,
                results,
                1,
            )

            hit3 = hit_at_k(
                expected,
                results,
                3,
            )

            hit5 = hit_at_k(
                expected,
                results,
                5,
            )

            rr = reciprocal_rank(
                expected,
                rank,
            )

            # ---------------------------------------------------------
            # Procedure detection
            # ---------------------------------------------------------

            detection_correct = (
                procedure_match(
                    expected,
                    detected_procedure,
                )
            )

            expected_none = (
                is_none_expected(
                    expected
                )
            )

            # ---------------------------------------------------------
            # Out-of-domain / ambiguous
            # ---------------------------------------------------------

            out_of_domain = (
                is_out_of_domain(row)
                or expected_none
            )

            ambiguous = (
                is_ambiguous_case(row)
            )

            if out_of_domain:
                # FIX: retrieve()/retrieve_with_diagnostics() deliberately
                # never return an empty result list as long as Qdrant has
                # any point at all (see select_candidates()/
                # filter_final_results() in retrieval.py - the pipeline
                # prefers returning the best available evidence over
                # returning nothing). Requiring len(results) == 0 here
                # made every out-of-domain case fail regardless of
                # retrieval quality, since that condition can now never be
                # true. What actually matters for an out-of-domain query
                # is that the system did NOT confidently lock onto a
                # specific procedure - i.e. the same bar already used for
                # the "ambiguous" category below.
                passed = (
                    detected_procedure is None
                )

            elif ambiguous:
                # For ambiguous cases, require the detector to stop
                # instead of confidently guessing.
                passed = (
                    detected_procedure is None
                    or hit5 == 1
                )

            else:
                passed = (
                    hit5 == 1
                )

            # ---------------------------------------------------------
            # Top scores
            # ---------------------------------------------------------

            top1_procedure = ""

            top1_final_score = 0.0
            top1_vector_score = 0.0

            if results:

                top1 = results[0]

                top1_procedure = (
                    top1.procedure
                )

                top1_final_score = (
                    float(
                        top1.final_score
                    )
                )

                top1_vector_score = (
                    float(
                        top1.vector_score
                    )
                )

                final_scores.append(
                    top1_final_score
                )

            # ---------------------------------------------------------
            # Print
            # ---------------------------------------------------------

            print(
                f"Intent             : "
                f"{detected_intent} "
                f"({intent_confidence:.3f})"
            )

            print(
                f"Detected procedure : "
                f"{detected_procedure}"
            )

            print(
                f"Expected procedure : "
                f"{expected}"
            )

            print(
                f"Top-1 procedure    : "
                f"{top1_procedure}"
            )

            print(
                f"Correct rank       : "
                f"{rank}"
            )

            print(
                f"Hit@1 / @3 / @5    : "
                f"{hit1} / {hit3} / {hit5}"
            )

            print(
                f"Detection correct  : "
                f"{int(detection_correct)}"
            )

            print(
                f"Benchmark pass     : "
                f"{'YES' if passed else 'NO'}"
            )

            print(
                f"Latency            : "
                f"{latency_ms:.2f} ms"
            )

            rows.append(
                {
                    "id": query_id,
                    "sheet": row.get(
                        "sheet",
                        "",
                    ),
                    "category": category,
                    "label": label,
                    "query": query,
                    "expected_procedure": expected,
                    "detected_intent": detected_intent,
                    "intent_confidence": intent_confidence,
                    "detected_procedure": detected_procedure or "",
                    "procedure_confidence": procedure_confidence,
                    "top1_procedure": top1_procedure,
                    "correct_rank": rank,
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                    "hit_at_5": hit5,
                    "mrr": rr,
                    "procedure_detection_correct": int(
                        detection_correct
                    ),
                    "out_of_domain": int(
                        out_of_domain
                    ),
                    "ambiguous": int(
                        ambiguous
                    ),
                    "benchmark_pass": int(
                        passed
                    ),
                    "top1_final_score": top1_final_score,
                    "top1_vector_score": top1_vector_score,
                    "result_count": len(results),
                    "latency_ms": latency_ms,
                }
            )

    finally:
        try:
            client.close()
        except Exception:
            pass

    # =============================================================================
    # DATAFRAME
    # =============================================================================

    results_df = pd.DataFrame(
        rows
    )

    # =============================================================================
    # OVERALL METRICS
    # =============================================================================

    total = len(
        results_df
    )

    pass_rate = (
        results_df["benchmark_pass"].mean()
        if total
        else 0.0
    )

    hit1 = (
        results_df["hit_at_1"].mean()
        if total
        else 0.0
    )

    hit3 = (
        results_df["hit_at_3"].mean()
        if total
        else 0.0
    )

    hit5 = (
        results_df["hit_at_5"].mean()
        if total
        else 0.0
    )

    mrr = (
        results_df["mrr"].mean()
        if total
        else 0.0
    )

    procedure_accuracy = (
        results_df[
            "procedure_detection_correct"
        ].mean()
        if total
        else 0.0
    )

    # Only queries with a concrete expected procedure
    # are meaningful for procedure detection accuracy.
    concrete = results_df[
        results_df[
            "expected_procedure"
        ].apply(
            lambda value:
                not is_none_expected(value)
        )
    ]

    procedure_accuracy_concrete = (
        concrete[
            "procedure_detection_correct"
        ].mean()
        if not concrete.empty
        else 0.0
    )

    # =============================================================================
    # LATENCY
    # =============================================================================

    latency_values = (
        results_df["latency_ms"]
        .astype(float)
        .tolist()
    )

    latency_mean = (
        statistics.mean(
            latency_values
        )
        if latency_values
        else 0.0
    )

    latency_p50 = percentile(
        latency_values,
        0.50,
    )

    latency_p95 = percentile(
        latency_values,
        0.95,
    )

    latency_max = (
        max(latency_values)
        if latency_values
        else 0.0
    )

    # =============================================================================
    # CATEGORY SUMMARY
    # =============================================================================

    category_rows = []

    for category, group in (
        results_df.groupby(
            "category",
            dropna=False,
        )
    ):

        category_rows.append(
            {
                "category": category,
                "queries": len(group),
                "pass_rate": group[
                    "benchmark_pass"
                ].mean(),
                "hit_at_1": group[
                    "hit_at_1"
                ].mean(),
                "hit_at_3": group[
                    "hit_at_3"
                ].mean(),
                "hit_at_5": group[
                    "hit_at_5"
                ].mean(),
                "mrr": group[
                    "mrr"
                ].mean(),
                "procedure_detection_accuracy": group[
                    "procedure_detection_correct"
                ].mean(),
                "avg_latency_ms": group[
                    "latency_ms"
                ].mean(),
            }
        )

    category_df = pd.DataFrame(
        category_rows
    )

    # =============================================================================
    # SUMMARY DATAFRAME
    # =============================================================================

    summary_df = pd.DataFrame(
        [
            {
                "metric": "Total Queries",
                "value": total,
            },
            {
                "metric": "Overall Pass Rate",
                "value": pass_rate,
            },
            {
                "metric": "Hit@1",
                "value": hit1,
            },
            {
                "metric": "Hit@3",
                "value": hit3,
            },
            {
                "metric": "Hit@5",
                "value": hit5,
            },
            {
                "metric": "MRR",
                "value": mrr,
            },
            {
                "metric": "Procedure Detection Accuracy",
                "value": procedure_accuracy,
            },
            {
                "metric": "Procedure Detection Accuracy (Concrete)",
                "value": procedure_accuracy_concrete,
            },
            {
                "metric": "Latency Mean (ms)",
                "value": latency_mean,
            },
            {
                "metric": "Latency P50 (ms)",
                "value": latency_p50,
            },
            {
                "metric": "Latency P95 (ms)",
                "value": latency_p95,
            },
            {
                "metric": "Latency Max (ms)",
                "value": latency_max,
            },
        ]
    )

    # =============================================================================
    # SAVE CSV
    # =============================================================================

    results_df.to_csv(
        DETAILED_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    category_df.to_csv(
        CATEGORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # =============================================================================
    # EXCEL REPORT
    # =============================================================================

    with pd.ExcelWriter(
        REPORT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        category_df.to_excel(
            writer,
            sheet_name="Category_Summary",
            index=False,
        )

        results_df.to_excel(
            writer,
            sheet_name="Detailed_Results",
            index=False,
        )

    # =============================================================================
    # LATENCY PLOT
    # =============================================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        latency_values,
        bins=30,
    )

    plt.xlabel(
        "Latency (ms)"
    )

    plt.ylabel(
        "Queries"
    )

    plt.title(
        "Retrieval Latency Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        LATENCY_PNG,
        dpi=150,
    )

    plt.close()

    # =============================================================================
    # FINAL SCORE PLOT
    # =============================================================================

    if final_scores:

        plt.figure(
            figsize=(10, 6)
        )

        plt.hist(
            final_scores,
            bins=30,
        )

        plt.xlabel(
            "Top-1 Final Score"
        )

        plt.ylabel(
            "Queries"
        )

        plt.title(
            "Top-1 Final Score Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            FINAL_SCORE_PNG,
            dpi=150,
        )

        plt.close()

    # =============================================================================
    # PRINT SUMMARY
    # =============================================================================

    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"Total queries                    : {total}"
    )

    print(
        f"Overall pass rate                : "
        f"{pass_rate:.2%}"
    )

    print(
        f"Hit@1                            : "
        f"{hit1:.2%}"
    )

    print(
        f"Hit@3                            : "
        f"{hit3:.2%}"
    )

    print(
        f"Hit@5                            : "
        f"{hit5:.2%}"
    )

    print(
        f"MRR                              : "
        f"{mrr:.4f}"
    )

    print(
        f"Procedure detection accuracy     : "
        f"{procedure_accuracy:.2%}"
    )

    print(
        f"Procedure detection (concrete)   : "
        f"{procedure_accuracy_concrete:.2%}"
    )

    print(
        f"Latency mean                     : "
        f"{latency_mean:.2f} ms"
    )

    print(
        f"Latency P50                      : "
        f"{latency_p50:.2f} ms"
    )

    print(
        f"Latency P95                      : "
        f"{latency_p95:.2f} ms"
    )

    print(
        f"Latency max                      : "
        f"{latency_max:.2f} ms"
    )

    print()
    print(
        f"Saved detailed results: "
        f"{DETAILED_CSV}"
    )

    print(
        f"Saved category summary: "
        f"{CATEGORY_CSV}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Saved Excel report: "
        f"{REPORT_XLSX}"
    )

    print(
        f"Saved latency plot: "
        f"{LATENCY_PNG}"
    )

    print(
        f"Saved final score plot: "
        f"{FINAL_SCORE_PNG}"
    )

    print("=" * 100)
    print("BENCHMARK COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()