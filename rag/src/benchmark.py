"""
LEGAL RAG - BENCHMARK

Benchmark:
    - Benchmark_84
    - Robustness_Queries

Dataset:
    data/benchmark/legal_rag_benchmark_84_plus_robustness.xlsx

Pipeline thực tế:

    Query
      ↓
    detect_intent()
      ↓
    detect_procedure()
      ↓
    Vector Retrieval
      ↓
    Lexical Retrieval
      ↓
    RRF Hybrid Fusion
      ↓
    Candidate Selection
      ↓
    Reranking
      ↓
    Final Filter
      ↓
    Final Top-K

QUAN TRỌNG:
    Benchmark gọi trực tiếp retrieve() trong retrieval.py.
    Không tự dựng lại pipeline retrieval.

Chạy từ:
    D:\\legal-rag

    python src/benchmark.py
"""

from __future__ import annotations

import contextlib
import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from retrieval import (
    connect_qdrant,
    detect_intent,
    detect_procedure,
    load_model,
    load_procedure_names,
    retrieve,
)


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


# =============================================================================
# SPECIAL LABELS
# =============================================================================

AMBIGUOUS = "AMBIGUOUS"
OUT_OF_DATASET = "OUT_OF_DATASET"


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(text: Any) -> str:
    """
    Normalize Vietnamese text only for benchmark comparison.

    Không thay đổi query gốc.
    Không thay đổi dữ liệu Qdrant.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()

    # Vietnamese đ
    text = text.replace("đ", "d")

    # Remove Vietnamese accents
    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # Keep only alphanumeric + spaces
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =============================================================================
# PROCEDURE MATCH
# =============================================================================

def procedure_match(
    expected: Any,
    actual: Any,
) -> bool:
    """
    Exact normalized procedure matching.

    Không cho partial credit:
        "Đăng ký kết hôn"
    không được coi là giống
        "Đăng ký kết hôn có yếu tố nước ngoài"
    """

    if expected is None:
        return False

    if actual is None:
        return False

    expected = str(expected).strip()
    actual = str(actual).strip()

    if not expected or not actual:
        return False

    special_values = {
        AMBIGUOUS,
        OUT_OF_DATASET,
        "NONE",
    }

    if expected.upper() in special_values:
        return False

    return (
        normalize_text(expected)
        == normalize_text(actual)
    )


# =============================================================================
# RESULT HELPERS
# =============================================================================

def get_result_procedure(
    result: Any,
) -> str:
    """
    Hỗ trợ cả:
        - RetrievalResult object
        - dict
    """

    # Nếu result là dict
    if isinstance(result, dict):

        data = result.get(
            "data",
            result,
        )

        return str(
            data.get(
                "procedure",
                "",
            )
        ).strip()

    # Nếu result là object
    return str(
        getattr(
            result,
            "procedure",
            "",
        )
    ).strip()


def get_result_score(
    result: Any,
) -> float:
    """
    Lấy final score an toàn.
    """

    if isinstance(result, dict):

        value = result.get(
            "final_score",
            result.get(
                "score",
                0.0,
            ),
        )

    else:

        value = getattr(
            result,
            "final_score",
            getattr(
                result,
                "score",
                0.0,
            ),
        )

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


# =============================================================================
# RUN ONE QUERY
# =============================================================================

def run_one_query(
    query: str,
    model: Any,
    client: Any,
    procedure_names: list[str],
) -> dict[str, Any]:
    """
    Chạy đúng pipeline hiện tại trong retrieval.py.

    Benchmark không truyền expected_procedure
    vào retrieval -> tránh leakage.
    """

    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # DETECT INTENT
    # -------------------------------------------------------------------------

    detected_intent, intent_confidence = (
        detect_intent(query)
    )

    # -------------------------------------------------------------------------
    # DETECT PROCEDURE
    # -------------------------------------------------------------------------

    detected_procedure, procedure_confidence = (
        detect_procedure(
            query,
            procedure_names,
            detected_intent,
        )
    )

    # -------------------------------------------------------------------------
    # CURRENT RETRIEVAL PIPELINE
    # -------------------------------------------------------------------------

    # retrieve() đã tự:
    #   detect intent
    #   detect procedure
    #   vector retrieval
    #   lexical retrieval
    #   RRF
    #   candidate selection
    #   rerank
    #   final filter

    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        final_results = retrieve(
            query=query,
            model=model,
            client=client,
            procedures=procedure_names,
        )

    # -------------------------------------------------------------------------
    # LATENCY
    # -------------------------------------------------------------------------

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return {
        "results": final_results,

        "intent": detected_intent,
        "intent_confidence": (
            float(intent_confidence)
        ),

        "detected_procedure": (
            detected_procedure
            if detected_procedure
            else ""
        ),

        "procedure_confidence": (
            float(procedure_confidence)
        ),

        "latency_ms": latency_ms,
    }


# =============================================================================
# EVALUATE ONE QUERY
# =============================================================================

def evaluate_one_query(
    item: pd.Series,
    output: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate:
        - Hit@1
        - Hit@3
        - Hit@5
        - MRR
        - ambiguity handling
        - out-of-dataset handling
    """

    query_id = int(
        item["id"]
    )

    query = str(
        item["query"]
    ).strip()

    expected_procedure = str(
        item["expected_procedure"]
    ).strip()

    expected_intent = str(
        item.get(
            "expected_intent",
            "",
        )
    ).strip()

    variant_type = str(
        item.get(
            "type",
            item.get(
                "variant_type",
                "",
            ),
        )
    ).strip()

    notes = str(
        item.get(
            "notes",
            item.get(
                "test_focus",
                "",
            ),
        )
    ).strip()

    # -------------------------------------------------------------------------
    # RETRIEVAL RESULTS
    # -------------------------------------------------------------------------

    results = output["results"]

    procedures = []

    for result in results:

        procedure = get_result_procedure(
            result
        )

        if procedure:
            procedures.append(
                procedure
            )

    # -------------------------------------------------------------------------
    # FIND CORRECT RANK
    # -------------------------------------------------------------------------

    correct_ranks = []

    for rank, procedure in enumerate(
        procedures,
        start=1,
    ):

        if procedure_match(
            expected_procedure,
            procedure,
        ):
            correct_ranks.append(
                rank
            )

    first_correct_rank = (
        min(correct_ranks)
        if correct_ranks
        else None
    )

    # -------------------------------------------------------------------------
    # HIT@K
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # SPECIAL CASES
    # -------------------------------------------------------------------------

    expected_ambiguous = (
        expected_procedure.upper()
        == AMBIGUOUS
    )

    expected_out_of_dataset = (
        expected_procedure.upper()
        == OUT_OF_DATASET
    )

    no_results = (
        len(results) == 0
    )

    detected_procedure = (
        output["detected_procedure"]
    )

    # -------------------------------------------------------------------------
    # AMBIGUOUS PASS
    #
    # Kỳ vọng:
    #   - không tự chọn procedure
    #   - không trả kết quả
    # -------------------------------------------------------------------------

    ambiguity_pass = int(
        expected_ambiguous
        and no_results
        and not detected_procedure
    )

    # -------------------------------------------------------------------------
    # OUT-OF-DATASET PASS
    #
    # Kỳ vọng:
    #   - không tự map sang procedure
    # -------------------------------------------------------------------------

    out_of_dataset_pass = int(
        expected_out_of_dataset
        and no_results
        and not detected_procedure
    )

    # -------------------------------------------------------------------------
    # NORMAL ON-TOPIC PASS
    # -------------------------------------------------------------------------

    normal_query = (
        not expected_ambiguous
        and not expected_out_of_dataset
    )

    normal_pass = int(
        normal_query
        and first_correct_rank is not None
    )

    # -------------------------------------------------------------------------
    # OVERALL BENCHMARK PASS
    # -------------------------------------------------------------------------

    if expected_ambiguous:

        benchmark_pass = ambiguity_pass

    elif expected_out_of_dataset:

        benchmark_pass = out_of_dataset_pass

    else:

        benchmark_pass = normal_pass

    # -------------------------------------------------------------------------
    # TOP 1
    # -------------------------------------------------------------------------

    top1_procedure = (
        procedures[0]
        if procedures
        else ""
    )

    top1_score = (
        get_result_score(
            results[0]
        )
        if results
        else 0.0
    )

    # -------------------------------------------------------------------------
    # RETURN
    # -------------------------------------------------------------------------

    return {
        "id": query_id,
        "query": query,

        "type": variant_type,

        "expected_procedure": (
            expected_procedure
        ),

        "expected_intent": (
            expected_intent
        ),

        "notes": notes,

        "detected_intent": (
            output["intent"]
        ),

        "intent_confidence": (
            output[
                "intent_confidence"
            ]
        ),

        "detected_procedure": (
            detected_procedure
        ),

        "procedure_confidence": (
            output[
                "procedure_confidence"
            ]
        ),

        "top1_procedure": (
            top1_procedure
        ),

        "top1_final_score": (
            top1_score
        ),

        "num_final_results": (
            len(results)
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

        "ambiguity_pass": (
            ambiguity_pass
        ),

        "out_of_dataset_pass": (
            out_of_dataset_pass
        ),

        "benchmark_pass": (
            benchmark_pass
        ),

        "latency_ms": (
            output["latency_ms"]
        ),
    }


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

def calculate_retrieval_metrics(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Hit@K + MRR.

    Chỉ tính trên query có expected procedure
    bình thường.
    """

    normal = df[
        ~df["expected_procedure"].isin(
            [
                AMBIGUOUS,
                OUT_OF_DATASET,
            ]
        )
    ].copy()

    n = len(normal)

    if n == 0:

        return {
            "count": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
        }

    return {
        "count": n,

        "hit_at_1": float(
            normal[
                "hit_at_1"
            ].mean()
        ),

        "hit_at_3": float(
            normal[
                "hit_at_3"
            ].mean()
        ),

        "hit_at_5": float(
            normal[
                "hit_at_5"
            ].mean()
        ),

        "mrr": float(
            normal[
                "reciprocal_rank"
            ].mean()
        ),
    }


# =============================================================================
# ROBUSTNESS METRICS
# =============================================================================

def calculate_robustness_metrics(
    df: pd.DataFrame,
) -> dict[str, Any]:

    ambiguous_df = df[
        df[
            "expected_procedure"
        ].eq(AMBIGUOUS)
    ]

    out_df = df[
        df[
            "expected_procedure"
        ].eq(OUT_OF_DATASET)
    ]

    ambiguous_rate = (
        float(
            ambiguous_df[
                "ambiguity_pass"
            ].mean()
        )
        if len(ambiguous_df)
        else 0.0
    )

    out_rate = (
        float(
            out_df[
                "out_of_dataset_pass"
            ].mean()
        )
        if len(out_df)
        else 0.0
    )

    return {
        "ambiguous_count": (
            len(ambiguous_df)
        ),

        "ambiguous_pass_rate": (
            ambiguous_rate
        ),

        "out_of_dataset_count": (
            len(out_df)
        ),

        "out_of_dataset_pass_rate": (
            out_rate
        ),
    }


# =============================================================================
# LATENCY
# =============================================================================

def calculate_latency(
    df: pd.DataFrame,
) -> dict[str, float]:

    values = df[
        "latency_ms"
    ]

    return {
        "mean_ms": float(
            values.mean()
        ),

        "p50_ms": float(
            values.quantile(
                0.50
            )
        ),

        "p95_ms": float(
            values.quantile(
                0.95
            )
        ),

        "max_ms": float(
            values.max()
        ),
    }


# =============================================================================
# CATEGORY SUMMARY
# =============================================================================

def build_category_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for category, group in df.groupby(
        "type",
        dropna=False,
    ):

        normal = group[
            ~group[
                "expected_procedure"
            ].isin(
                [
                    AMBIGUOUS,
                    OUT_OF_DATASET,
                ]
            )
        ]

        rows.append(
            {
                "category": category,

                "queries": len(group),

                "pass_rate": float(
                    group[
                        "benchmark_pass"
                    ].mean()
                ),

                "hit_at_1": (
                    float(
                        normal[
                            "hit_at_1"
                        ].mean()
                    )
                    if len(normal)
                    else None
                ),

                "hit_at_3": (
                    float(
                        normal[
                            "hit_at_3"
                        ].mean()
                    )
                    if len(normal)
                    else None
                ),

                "hit_at_5": (
                    float(
                        normal[
                            "hit_at_5"
                        ].mean()
                    )
                    if len(normal)
                    else None
                ),

                "mrr": (
                    float(
                        normal[
                            "reciprocal_rank"
                        ].mean()
                    )
                    if len(normal)
                    else None
                ),

                "avg_latency_ms": float(
                    group[
                        "latency_ms"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "category"
    )


# =============================================================================
# PLOTS
# =============================================================================

def save_latency_plot(
    df: pd.DataFrame,
) -> Path:

    path = (
        RESULT_DIR
        / "latency_distribution.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df["latency_ms"],
        bins=20,
    )

    plt.xlabel(
        "Latency (ms)"
    )

    plt.ylabel(
        "Number of queries"
    )

    plt.title(
        "Benchmark Latency Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


def save_score_plot(
    df: pd.DataFrame,
) -> Path:

    path = (
        RESULT_DIR
        / "final_score_distribution.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    normal = df[
        ~df[
            "expected_procedure"
        ].isin(
            [
                AMBIGUOUS,
                OUT_OF_DATASET,
            ]
        )
    ]

    special = df[
        df[
            "expected_procedure"
        ].isin(
            [
                AMBIGUOUS,
                OUT_OF_DATASET,
            ]
        )
    ]

    if len(normal):

        plt.hist(
            normal[
                "top1_final_score"
            ],
            bins=15,
            alpha=0.7,
            label="On-topic",
        )

    if len(special):

        plt.hist(
            special[
                "top1_final_score"
            ],
            bins=15,
            alpha=0.7,
            label="Ambiguous / Out-of-dataset",
        )

    plt.xlabel(
        "Top-1 Final Score"
    )

    plt.ylabel(
        "Number of queries"
    )

    plt.title(
        "Final Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    return path


# =============================================================================
# LOAD BENCHMARK EXCEL
# =============================================================================

def load_benchmark_excel() -> pd.DataFrame:
    """
    Đọc cả 2 sheet:

        Benchmark_84
        Robustness_Queries

    Có thể tự bỏ qua sheet không có cột query.
    """

    workbook = pd.ExcelFile(
        BENCHMARK_FILE
    )

    print()
    print(
        f"Excel sheets: "
        f"{workbook.sheet_names}"
    )

    frames = []

    for sheet_name in (
        workbook.sheet_names
    ):

        df = pd.read_excel(
            BENCHMARK_FILE,
            sheet_name=sheet_name,
        )

        if "query" not in df.columns:
            continue

        df = df.copy()

        # Robustness sheet dùng variant_type.
        if (
            "variant_type"
            in df.columns
        ):

            df[
                "type"
            ] = df[
                "variant_type"
            ]

        # Robustness sheet dùng test_focus.
        if (
            "test_focus"
            in df.columns
        ):

            df[
                "notes"
            ] = df[
                "test_focus"
            ]

        frames.append(df)

    if not frames:

        raise ValueError(
            "Không tìm thấy sheet benchmark "
            "có cột 'query'."
        )

    benchmark = pd.concat(
        frames,
        ignore_index=True,
    )

    required_columns = {
        "id",
        "query",
        "expected_procedure",
    }

    missing_columns = (
        required_columns
        - set(
            benchmark.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Thiếu cột benchmark: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if "type" not in benchmark.columns:

        benchmark["type"] = (
            "on_topic"
        )

    if "notes" not in benchmark.columns:

        benchmark["notes"] = ""

    if (
        "expected_intent"
        not in benchmark.columns
    ):

        benchmark[
            "expected_intent"
        ] = ""

    # Clean
    benchmark[
        "query"
    ] = (
        benchmark[
            "query"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    benchmark = benchmark[
        benchmark[
            "query"
        ].ne("")
    ].copy()

    # Avoid accidental duplicate rows.
    benchmark = benchmark.drop_duplicates(
        subset=[
            "id",
            "query",
        ],
        keep="first",
    )

    return benchmark


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 90)
    print(
        "VIETNAMESE LEGAL RAG - "
        "HYBRID RETRIEVAL BENCHMARK"
    )
    print("=" * 90)

    # -------------------------------------------------------------------------
    # RESULT DIRECTORY
    # -------------------------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # LOAD BENCHMARK
    # -------------------------------------------------------------------------

    if not BENCHMARK_FILE.exists():

        raise FileNotFoundError(
            "\nKhông tìm thấy benchmark file:\n"
            f"{BENCHMARK_FILE}\n\n"
            "Hãy kiểm tra file có đúng ở:\n"
            f"{BASE_DIR / 'data' / 'benchmark'}"
        )

    benchmark = (
        load_benchmark_excel()
    )

    total_queries = len(
        benchmark
    )

    print()
    print(
        f"Benchmark file : "
        f"{BENCHMARK_FILE}"
    )

    print(
        f"Total queries  : "
        f"{total_queries}"
    )

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    print()
    print(
        "Loading embedding model..."
    )

    model = load_model()

    # -------------------------------------------------------------------------
    # CONNECT QDRANT
    # -------------------------------------------------------------------------

    print()
    print(
        "Connecting Qdrant..."
    )

    client = connect_qdrant()

    try:

        # ---------------------------------------------------------------------
        # LOAD PROCEDURES
        # ---------------------------------------------------------------------

        procedure_names = (
            load_procedure_names(
                client
            )
        )

        print(
            f"Procedure names: "
            f"{len(procedure_names)}"
        )

        # ---------------------------------------------------------------------
        # RUN
        # ---------------------------------------------------------------------

        rows = []

        for index, item in benchmark.iterrows():

            query = str(
                item["query"]
            ).strip()

            print()
            print("-" * 90)

            print(
                f"[{index + 1:03d}/"
                f"{total_queries:03d}] "
                f"{query}"
            )

            output = run_one_query(
                query=query,
                model=model,
                client=client,
                procedure_names=(
                    procedure_names
                ),
            )

            evaluated = (
                evaluate_one_query(
                    item=item,
                    output=output,
                )
            )

            rows.append(
                evaluated
            )

            print(
                f"Intent            : "
                f"{evaluated['detected_intent']} "
                f"("
                f"{evaluated['intent_confidence']:.3f}"
                f")"
            )

            print(
                f"Detected procedure: "
                f"{evaluated['detected_procedure']}"
            )

            print(
                f"Top-1 procedure   : "
                f"{evaluated['top1_procedure']}"
            )

            print(
                f"Correct rank      : "
                f"{evaluated['first_correct_rank']}"
            )

            print(
                f"Hit@1             : "
                f"{evaluated['hit_at_1']}"
            )

            print(
                f"Hit@3             : "
                f"{evaluated['hit_at_3']}"
            )

            print(
                f"Hit@5             : "
                f"{evaluated['hit_at_5']}"
            )

            print(
                f"Benchmark pass    : "
                f"{'YES' if evaluated['benchmark_pass'] else 'NO'}"
            )

            print(
                f"Latency           : "
                f"{evaluated['latency_ms']:.2f} ms"
            )

    finally:

        client.close()

    # -------------------------------------------------------------------------
    # DATAFRAME
    # -------------------------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    retrieval_metrics = (
        calculate_retrieval_metrics(
            results_df
        )
    )

    robustness_metrics = (
        calculate_robustness_metrics(
            results_df
        )
    )

    latency_metrics = (
        calculate_latency(
            results_df
        )
    )

    overall_pass_rate = float(
        results_df[
            "benchmark_pass"
        ].mean()
    )

    # -------------------------------------------------------------------------
    # CATEGORY
    # -------------------------------------------------------------------------

    category_df = (
        build_category_summary(
            results_df
        )
    )

    # -------------------------------------------------------------------------
    # SAVE CSV
    # -------------------------------------------------------------------------

    detailed_csv = (
        RESULT_DIR
        / "benchmark_detailed_results.csv"
    )

    category_csv = (
        RESULT_DIR
        / "benchmark_category_summary.csv"
    )

    summary_csv = (
        RESULT_DIR
        / "benchmark_summary.csv"
    )

    results_df.to_csv(
        detailed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    category_df.to_csv(
        category_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # SUMMARY DATA
    # -------------------------------------------------------------------------

    summary_rows = [
        {
            "metric": "Total Queries",
            "value": total_queries,
        },

        {
            "metric": "Overall Pass Rate",
            "value": overall_pass_rate,
        },

        {
            "metric": "Retrieval Query Count",
            "value": retrieval_metrics[
                "count"
            ],
        },

        {
            "metric": "Hit@1",
            "value": retrieval_metrics[
                "hit_at_1"
            ],
        },

        {
            "metric": "Hit@3",
            "value": retrieval_metrics[
                "hit_at_3"
            ],
        },

        {
            "metric": "Hit@5",
            "value": retrieval_metrics[
                "hit_at_5"
            ],
        },

        {
            "metric": "MRR",
            "value": retrieval_metrics[
                "mrr"
            ],
        },

        {
            "metric": "Ambiguous Count",
            "value": robustness_metrics[
                "ambiguous_count"
            ],
        },

        {
            "metric": "Ambiguous Pass Rate",
            "value": robustness_metrics[
                "ambiguous_pass_rate"
            ],
        },

        {
            "metric": "Out-of-Dataset Count",
            "value": robustness_metrics[
                "out_of_dataset_count"
            ],
        },

        {
            "metric": "Out-of-Dataset Pass Rate",
            "value": robustness_metrics[
                "out_of_dataset_pass_rate"
            ],
        },

        {
            "metric": "Latency Mean (ms)",
            "value": latency_metrics[
                "mean_ms"
            ],
        },

        {
            "metric": "Latency P50 (ms)",
            "value": latency_metrics[
                "p50_ms"
            ],
        },

        {
            "metric": "Latency P95 (ms)",
            "value": latency_metrics[
                "p95_ms"
            ],
        },

        {
            "metric": "Latency Max (ms)",
            "value": latency_metrics[
                "max_ms"
            ],
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # EXCEL REPORT
    # -------------------------------------------------------------------------

    report_xlsx = (
        RESULT_DIR
        / "benchmark_report.xlsx"
    )

    with pd.ExcelWriter(
        report_xlsx,
        engine="openpyxl",
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Detailed",
            index=False,
        )

        category_df.to_excel(
            writer,
            sheet_name="Categories",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        for ws in (
            writer.book.worksheets
        ):

            ws.freeze_panes = "A2"

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------

    latency_plot = (
        save_latency_plot(
            results_df
        )
    )

    score_plot = (
        save_score_plot(
            results_df
        )
    )

    # -------------------------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # -------------------------------------------------------------------------

    print()
    print()
    print("=" * 90)
    print(
        "BENCHMARK SUMMARY"
    )
    print("=" * 90)

    print(
        f"Total queries            : "
        f"{total_queries}"
    )

    print(
        f"Overall pass rate        : "
        f"{overall_pass_rate:.2%}"
    )

    print()
    print(
        f"Hit@1                    : "
        f"{retrieval_metrics['hit_at_1']:.2%}"
    )

    print(
        f"Hit@3                    : "
        f"{retrieval_metrics['hit_at_3']:.2%}"
    )

    print(
        f"Hit@5                    : "
        f"{retrieval_metrics['hit_at_5']:.2%}"
    )

    print(
        f"MRR                      : "
        f"{retrieval_metrics['mrr']:.4f}"
    )

    print()
    print(
        f"Ambiguous pass rate      : "
        f"{robustness_metrics['ambiguous_pass_rate']:.2%}"
    )

    print(
        f"Out-of-dataset pass rate : "
        f"{robustness_metrics['out_of_dataset_pass_rate']:.2%}"
    )

    print()
    print(
        f"Latency mean             : "
        f"{latency_metrics['mean_ms']:.2f} ms"
    )

    print(
        f"Latency P50              : "
        f"{latency_metrics['p50_ms']:.2f} ms"
    )

    print(
        f"Latency P95              : "
        f"{latency_metrics['p95_ms']:.2f} ms"
    )

    print(
        f"Latency max              : "
        f"{latency_metrics['max_ms']:.2f} ms"
    )

    # -------------------------------------------------------------------------
    # CATEGORY TABLE
    # -------------------------------------------------------------------------

    print()
    print("-" * 90)
    print(
        "CATEGORY SUMMARY"
    )
    print("-" * 90)

    print(
        category_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # FILES
    # -------------------------------------------------------------------------

    print()
    print("-" * 90)
    print(
        "FILES GENERATED"
    )
    print("-" * 90)

    print(
        f"- {detailed_csv}"
    )

    print(
        f"- {category_csv}"
    )

    print(
        f"- {summary_csv}"
    )

    print(
        f"- {report_xlsx}"
    )

    print(
        f"- {latency_plot}"
    )

    print(
        f"- {score_plot}"
    )

    print()
    print("=" * 90)
    print(
        "BENCHMARK COMPLETED"
    )
    print("=" * 90)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[INFO] Benchmark interrupted."
        )

    except Exception as exc:

        print()
        print("=" * 90)
        print(
            "[FATAL ERROR]"
        )
        print("=" * 90)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 90)

        raise