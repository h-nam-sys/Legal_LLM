from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = Path(
    r"D:/legal-rag/data/raw/administrative_procedures.xlsx"
)


# ============================================================
# MAIN
# ============================================================

def inspect_excel():
    print("=" * 80)
    print("VIETNAMESE ADMINISTRATIVE PROCEDURES")
    print("EXCEL DATA INSPECTION")
    print("=" * 80)

    print()
    print(f"Input file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nKhông tìm thấy file:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 1. Excel workbook
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("1. WORKBOOK")
    print("-" * 80)

    excel = pd.ExcelFile(INPUT_FILE)

    print(f"Number of sheets: {len(excel.sheet_names)}")

    print()
    print("Sheets:")

    for i, sheet in enumerate(excel.sheet_names, start=1):
        print(f"{i}. {sheet}")

    # --------------------------------------------------------
    # 2. Inspect every sheet
    # --------------------------------------------------------

    for sheet_name in excel.sheet_names:

        print()
        print("=" * 80)
        print(f"SHEET: {sheet_name}")
        print("=" * 80)

        try:
            df = pd.read_excel(
                INPUT_FILE,
                sheet_name=sheet_name
            )

        except Exception as exc:
            print(f"[ERROR] Cannot read sheet: {exc}")
            continue

        print()
        print(f"Rows    : {len(df):,}")
        print(f"Columns : {len(df.columns)}")

        print()
        print("Column names:")

        for i, column in enumerate(df.columns, start=1):
            print(f"{i}. {repr(column)}")

        # ----------------------------------------------------
        # Data types
        # ----------------------------------------------------

        print()
        print("Data types:")

        print(df.dtypes.to_string())

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        print()
        print("Missing values:")

        missing = df.isna().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if missing.empty:
            print("No missing values.")

        else:
            print(missing.to_string())

        # ----------------------------------------------------
        # Sample rows
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("FIRST 5 ROWS")
        print("-" * 80)

        pd.set_option(
            "display.max_columns",
            None
        )

        pd.set_option(
            "display.max_colwidth",
            100
        )

        print(
            df.head(5).to_string()
        )

        # ----------------------------------------------------
        # Non-null sample for object columns
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("NON-NULL SAMPLE VALUES")
        print("-" * 80)

        for column in df.columns:

            non_null = (
                df[column]
                .dropna()
                .astype(str)
            )

            if non_null.empty:
                continue

            print()
            print(f"[{column}]")

            for value in non_null.head(3):
                value = value.replace(
                    "\n",
                    " "
                )

                if len(value) > 300:
                    value = value[:300] + "..."

                print(f"  {value}")

    print()
    print("=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    inspect_excel()