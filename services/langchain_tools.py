from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from services.bi_tools import (
    get_dataset_schema,
    get_dataset_summary,
    calculate_total,
    calculate_average,
    calculate_count,
    group_by,
    get_top_n,
    get_time_trend,
    cross_dataset_analysis,
)


def build_bi_tools(selected_datasets):
    """
    Build LangChain tools for the datasets currently selected
    by the user.

    The tools can ONLY access these selected datasets.
    """

    dataset_map = {
        dataset.id: dataset
        for dataset in selected_datasets
    }

    allowed_dataset_ids = list(dataset_map.keys())

    # ========================================================
    # Helper
    # ========================================================

    def get_dataset(dataset_id):
        """
        Resolve a dataset ID only if it belongs to the
        user's currently selected datasets.
        """

        try:
            dataset_id = int(dataset_id)
        except (TypeError, ValueError):

            raise ValueError(
                "dataset_id must be a valid integer."
            )

        dataset = dataset_map.get(dataset_id)

        if dataset is None:

            raise ValueError(
                f"Dataset {dataset_id} is not available "
                f"in the current selection. "
                f"Available dataset IDs: "
                f"{allowed_dataset_ids}"
            )

        return dataset

    # ========================================================
    # TOOL 1 — Dataset Schema
    # ========================================================

    @tool
    def dataset_schema(dataset_id: int):
        """
        Get the columns, data types, row count and basic
        structure of one selected dataset.

        Use this tool when you need to understand what
        columns are actually available before performing
        a calculation.
        """

        dataset = get_dataset(dataset_id)

        return get_dataset_schema(dataset)

    # ========================================================
    # TOOL 2 — Dataset Summary
    # ========================================================

    @tool
    def dataset_summary(dataset_id: int):
        """
        Get an automatic summary of one selected dataset.

        This identifies numeric columns, categorical columns,
        date columns and likely business metrics such as
        revenue, sales, quantity and cost.
        """

        dataset = get_dataset(dataset_id)

        return get_dataset_summary(dataset)

    # ========================================================
    # TOOL 3 — Total
    # ========================================================

    @tool
    def total_value(
        dataset_id: int,
        column: str
    ):
        """
        Calculate the exact total/sum of a numeric column.

        IMPORTANT:
        The column must exist in the selected dataset.
        """

        dataset = get_dataset(dataset_id)

        try:

            return calculate_total(
                dataset,
                column
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    # ========================================================
    # TOOL 4 — Average
    # ========================================================

    @tool
    def average_value(
        dataset_id: int,
        column: str
    ):
        """
        Calculate the exact average of a numeric column.
        """

        dataset = get_dataset(dataset_id)

        try:

            return calculate_average(
                dataset,
                column
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    # ========================================================
    # TOOL 5 — Count
    # ========================================================

    @tool
    def count_records(
        dataset_id: int,
        column: Optional[str] = None
    ):
        """
        Count records in a selected dataset.

        If column is supplied, count non-null values
        in that column.

        If column is omitted, count all rows.
        """

        dataset = get_dataset(dataset_id)

        try:

            return calculate_count(
                dataset,
                column
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    # ========================================================
    # TOOL 6 — Group By
    # ========================================================

    @tool
    def grouped_analysis(
        dataset_id: int,
        dimension: str,
        metric: Optional[str] = None,
        aggregation: str = "sum"
    ):
        """
        Group a dataset by a dimension and calculate an
        aggregation.

        Supported aggregations:
        sum, mean, count, min, max

        Example:
        revenue by site
        quantity by item
        record count by city
        """

        dataset = get_dataset(dataset_id)

        try:

            return group_by(
                dataset=dataset,
                dimension=dimension,
                metric=metric,
                aggregation=aggregation
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    # ========================================================
    # TOOL 7 — Top N
    # ========================================================

    @tool
    def top_n_analysis(
        dataset_id: int,
        dimension: str,
        metric: str,
        n: int = 5
    ):
        """
        Find the top N categories, sites, products or other
        dimension values based on the sum of a numeric metric.
        """

        dataset = get_dataset(dataset_id)

        try:

            return get_top_n(
                dataset=dataset,
                dimension=dimension,
                metric=metric,
                n=n
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    # ========================================================
    # TOOL 8 — Time Trend
    # ========================================================

    @tool
    def time_trend_analysis(
        dataset_id: int,
        date_column: str,
        metric: str,
        aggregation: str = "sum",
        grain: str = "month"
    ):
        """
        Calculate a time-based trend.

        Supported grain:
        day, week, month, year

        Example:
        monthly revenue
        daily quantity
        yearly sales
        """

        dataset = get_dataset(dataset_id)

        try:

            return get_time_trend(
                dataset=dataset,
                date_column=date_column,
                metric=metric,
                aggregation=aggregation,
                grain=grain
            )

        except ValueError as error:

            schema = get_dataset_schema(dataset)

            return {
                "success": False,
                "error": str(error),
                "available_columns": [
                    item["name"]
                    for item in schema["column_details"]
                ]
            }

    @tool
    def cross_dataset_lookup(
        primary_dataset_id: int,
        primary_dimension: str,
        primary_metric: str,
        lookup_dataset_id: int,
        lookup_key: str,
        return_column: str,
        aggregation: str = "sum",
        ranking: str = "highest",
        n: int = 1,):
        """
        Analyze a metric in one selected dataset and then retrieve
        a related value from another selected dataset.

        Use this for questions that require both:
        1. Finding an important entity through aggregation/ranking.
        2. Looking up related information about that entity in another dataset.

        Example:
        Find the center associated with the site having the highest
        net revenue.
        """

        primary_dataset = dataset_map.get(primary_dataset_id)
        lookup_dataset = dataset_map.get(lookup_dataset_id)

        if primary_dataset is None:
            return {
                "success": False,
                "error": (
                    f"Primary dataset ID {primary_dataset_id} "
                    "is not selected."
                ),
            }

        if lookup_dataset is None:
            return {
                "success": False,
                "error": (
                    f"Lookup dataset ID {lookup_dataset_id} "
                    "is not selected."
                ),
            }

        try:

            return cross_dataset_analysis(
                primary_dataset=primary_dataset,
                primary_dimension=primary_dimension,
                primary_metric=primary_metric,
                lookup_dataset=lookup_dataset,
                lookup_key=lookup_key,
                return_column=return_column,
                aggregation=aggregation,
                ranking=ranking,
                n=n,
            )

        except ValueError as error:

            return {
                "success": False,
                "error": str(error),
            }

        except Exception as error:

            return {
                "success": False,
                "error": (
                    f"Cross-dataset analysis failed: {error}"
                ),
            }

    # ========================================================
    # Return all tools
    # ========================================================

    return [
        dataset_schema,
        dataset_summary,
        total_value,
        average_value,
        count_records,
        grouped_analysis,
        top_n_analysis,
        time_trend_analysis,
        cross_dataset_lookup,
    ]


    