from models.dataset import Dataset
from services.bi_tools import (
    get_dataset_schema,
    get_dataset_summary,
    calculate_count,
    group_by,
    get_top_n,
    get_time_trend
)

from app import app


with app.app_context():

    datasets = Dataset.query.all()

    if not datasets:
        print("No datasets found.")
        raise SystemExit

    print("\n===== AVAILABLE DATASETS =====")

    for dataset in datasets:
        print(
            f"ID: {dataset.id} | "
            f"Name: {dataset.original_filename} | "
            f"Rows: {dataset.row_count} | "
            f"Columns: {dataset.column_count}"
        )

    # ---------------------------------------------------------
    # Choose the dataset that actually has numeric data.
    #
    # Prefer sample-raffi.xlsx if it exists.
    # Otherwise use the first dataset.
    # ---------------------------------------------------------

    dataset = next(
        (
            d for d in datasets
            if d.original_filename
            and "raffi" in d.original_filename.lower()
        ),
        datasets[0]
    )

    print("\n========================================")
    print("TESTING DATASET:")
    print(dataset.original_filename)
    print("========================================")

    # ---------------------------------------------------------
    # SCHEMA
    # ---------------------------------------------------------

    print("\n===== SCHEMA =====")

    schema = get_dataset_schema(dataset)

    print(schema)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n===== SUMMARY =====")

    summary = get_dataset_summary(dataset)

    print(summary)

    # ---------------------------------------------------------
    # COUNT
    # ---------------------------------------------------------

    print("\n===== ROW COUNT =====")

    count_result = calculate_count(dataset)

    print(count_result)

    # ---------------------------------------------------------
    # DETECTED METRICS
    # ---------------------------------------------------------

    detected_metrics = summary["detected_metrics"]

    revenue_column = detected_metrics.get("revenue")
    quantity_column = detected_metrics.get("quantity")

    dimension_column = summary.get(
        "recommended_dimension"
    )

    date_column = summary.get(
        "date_column"
    )

    # ---------------------------------------------------------
    # TOTAL REVENUE / SALES
    # ---------------------------------------------------------

    if revenue_column:

        print(
            f"\n===== TOTAL {revenue_column.upper()} ====="
        )

        from services.bi_tools import calculate_total

        total_result = calculate_total(
            dataset,
            revenue_column
        )

        print(total_result)

    else:

        print(
            "\n===== TOTAL ====="
        )

        print(
            "No revenue/sales/amount/net numeric column "
            "was detected."
        )

    # ---------------------------------------------------------
    # AVERAGE REVENUE / SALES
    # ---------------------------------------------------------

    if revenue_column:

        print(
            f"\n===== AVERAGE {revenue_column.upper()} ====="
        )

        from services.bi_tools import calculate_average

        average_result = calculate_average(
            dataset,
            revenue_column
        )

        print(average_result)

    # ---------------------------------------------------------
    # GROUP BY
    # ---------------------------------------------------------

    if dimension_column and revenue_column:

        print(
            f"\n===== {revenue_column.upper()} "
            f"BY {dimension_column.upper()} ====="
        )

        grouped_result = group_by(
            dataset,
            dimension_column,
            revenue_column,
            "sum"
        )

        print(grouped_result)

    else:

        print(
            "\n===== GROUP BY ====="
        )

        print(
            "A suitable dimension + numeric metric "
            "was not detected."
        )

    # ---------------------------------------------------------
    # TOP N
    # ---------------------------------------------------------

    if dimension_column and revenue_column:

        print(
            f"\n===== TOP 5 "
            f"{dimension_column.upper()} ====="
        )

        top_result = get_top_n(
            dataset,
            dimension_column,
            revenue_column,
            5
        )

        print(top_result)

    # ---------------------------------------------------------
    # TIME TREND
    # ---------------------------------------------------------

    if date_column and revenue_column:

        print(
            f"\n===== TIME TREND ====="
        )

        trend_result = get_time_trend(
            dataset,
            date_column,
            revenue_column,
            "sum",
            "month"
        )

        print(trend_result)

    else:

        print(
            "\n===== TIME TREND ====="
        )

        print(
            "A date column and numeric metric "
            "were not detected."
        )