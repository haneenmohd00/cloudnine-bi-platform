from __future__ import annotations

from typing import Any, Optional

import pandas as pd


# ============================================================
# Helper functions
# ============================================================

def _load_dataset(dataset) -> pd.DataFrame:
    """
    Load a Dataset model object into a pandas DataFrame.

    This function supports the current project structure where
    the Dataset model stores the physical file location.
    """

    if dataset is None:
        raise ValueError("No dataset was provided.")

    file_path = getattr(dataset, "file_path", None)

    if not file_path:
        file_path = getattr(dataset, "stored_filename", None)

    if not file_path:
        raise ValueError(
            f"Could not determine the file path for dataset "
            f"'{getattr(dataset, 'original_filename', 'unknown')}'."
        )

    if str(file_path).lower().endswith(".csv"):
        return pd.read_csv(file_path)

    if str(file_path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported dataset format: {file_path}")


def _find_column(
    df: pd.DataFrame,
    requested_column: str
) -> Optional[str]:
    """
    Find a real dataframe column using case-insensitive matching.
    """

    if not requested_column:
        return None

    requested = requested_column.strip().lower()

    for column in df.columns:
        if str(column).strip().lower() == requested:
            return column

    return None


def _detect_numeric_column(
    df: pd.DataFrame,
    keywords: list[str]
) -> Optional[str]:
    """
    Find a numeric column based on semantic keywords.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    for column in numeric_columns:
        column_name = str(column).lower()

        for keyword in keywords:
            if keyword in column_name:
                return column

    return None


def _detect_date_column(
    df: pd.DataFrame
) -> Optional[str]:
    """
    Detect a likely date column.
    """

    date_keywords = [
        "date",
        "time",
        "datetime",
        "created",
        "timestamp"
    ]

    for column in df.columns:
        name = str(column).lower()

        if any(keyword in name for keyword in date_keywords):
            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if converted.notna().sum() > 0:
                return column

    return None


def _detect_dimension_column(
    df: pd.DataFrame
) -> Optional[str]:
    """
    Detect a useful categorical/dimension column.
    """

    preferred_keywords = [
        "site",
        "branch",
        "location",
        "category",
        "group",
        "product",
        "item",
        "department",
        "specialization",
        "physician"
    ]

    object_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # First try semantically meaningful columns.
    for keyword in preferred_keywords:
        for column in object_columns:
            if keyword in str(column).lower():
                return column

    # Otherwise use a reasonable categorical column.
    for column in object_columns:
        unique_count = df[column].nunique(dropna=True)

        if 1 < unique_count <= 50:
            return column

    return None


# ============================================================
# TOOL 1 — Dataset Schema
# ============================================================

def get_dataset_schema(dataset) -> dict[str, Any]:
    """
    Return the schema and basic information about a dataset.
    """

    df = _load_dataset(dataset)

    columns = []

    for column in df.columns:
        columns.append({
            "name": str(column),
            "dtype": str(df[column].dtype),
            "non_null_count": int(df[column].notna().sum()),
            "unique_values": int(df[column].nunique(dropna=True))
        })

    return {
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_details": columns
    }


# ============================================================
# TOOL 2 — Total / Sum
# ============================================================

def calculate_total(
    dataset,
    column: str
) -> dict[str, Any]:
    """
    Calculate the total/sum of a numeric column.
    """

    df = _load_dataset(dataset)

    real_column = _find_column(df, column)

    if real_column is None:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    numeric_values = pd.to_numeric(
        df[real_column],
        errors="coerce"
    )

    total = numeric_values.sum()

    return {
        "operation": "sum",
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "column": str(real_column),
        "value": float(total)
    }


# ============================================================
# TOOL 3 — Average
# ============================================================

def calculate_average(
    dataset,
    column: str
) -> dict[str, Any]:
    """
    Calculate the average of a numeric column.
    """

    df = _load_dataset(dataset)

    real_column = _find_column(df, column)

    if real_column is None:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    numeric_values = pd.to_numeric(
        df[real_column],
        errors="coerce"
    )

    average = numeric_values.mean()

    return {
        "operation": "average",
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "column": str(real_column),
        "value": float(average)
        if pd.notna(average)
        else None
    }


# ============================================================
# TOOL 4 — Count
# ============================================================

def calculate_count(
    dataset,
    column: Optional[str] = None
) -> dict[str, Any]:
    """
    Count rows or non-null values in a column.
    """

    df = _load_dataset(dataset)

    if column:
        real_column = _find_column(df, column)

        if real_column is None:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        value = int(df[real_column].notna().sum())

    else:
        value = int(len(df))

    return {
        "operation": "count",
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "column": column,
        "value": value
    }


# ============================================================
# TOOL 5 — Group By
# ============================================================

def group_by(
    dataset,
    dimension: str,
    metric: Optional[str] = None,
    aggregation: str = "sum"
) -> dict[str, Any]:
    """
    Group a dataset by a dimension and aggregate a metric.

    Supported aggregations:

        sum
        mean
        count
        min
        max

    For count, metric can be omitted.
    """

    df = _load_dataset(dataset)

    real_dimension = _find_column(
        df,
        dimension
    )

    if real_dimension is None:
        raise ValueError(
            f"Dimension '{dimension}' does not exist."
        )

    aggregation = aggregation.lower().strip()

    # ---------------------------------------------------------
    # COUNT
    #
    # Count rows for each dimension value.
    # No numeric metric is required.
    # ---------------------------------------------------------

    if aggregation == "count":

        result = (
            df.groupby(real_dimension)
            .size()
            .sort_values(ascending=False)
        )

        records = []

        for dimension_value, count in result.items():

            records.append({
                "dimension": str(dimension_value),
                "value": int(count)
            })

        return {
            "operation": "group_by",
            "dataset": getattr(
                dataset,
                "original_filename",
                "unknown"
            ),
            "dimension": str(real_dimension),
            "metric": None,
            "aggregation": "count",
            "results": records
        }

    # ---------------------------------------------------------
    # Other aggregations require a metric.
    # ---------------------------------------------------------

    if not metric:
        raise ValueError(
            f"Metric is required for '{aggregation}' aggregation."
        )

    real_metric = _find_column(
        df,
        metric
    )

    if real_metric is None:
        raise ValueError(
            f"Metric '{metric}' does not exist."
        )

    # ---------------------------------------------------------
    # Convert metric to numeric.
    # ---------------------------------------------------------

    df[real_metric] = pd.to_numeric(
        df[real_metric],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    if aggregation == "sum":

        result = (
            df.groupby(real_dimension)[real_metric]
            .sum()
            .sort_values(ascending=False)
        )

    elif aggregation == "mean":

        result = (
            df.groupby(real_dimension)[real_metric]
            .mean()
            .sort_values(ascending=False)
        )

    elif aggregation == "min":

        result = (
            df.groupby(real_dimension)[real_metric]
            .min()
            .sort_values(ascending=False)
        )

    elif aggregation == "max":

        result = (
            df.groupby(real_dimension)[real_metric]
            .max()
            .sort_values(ascending=False)
        )

    else:

        raise ValueError(
            "Unsupported aggregation. "
            "Use sum, mean, count, min, or max."
        )

    # ---------------------------------------------------------
    # Convert result into JSON-safe records
    # ---------------------------------------------------------

    records = []

    for dimension_value, metric_value in result.items():

        records.append({
            "dimension": str(dimension_value),
            "value": (
                float(metric_value)
                if pd.notna(metric_value)
                else None
            )
        })

    return {
        "operation": "group_by",
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "dimension": str(real_dimension),
        "metric": str(real_metric),
        "aggregation": aggregation,
        "results": records
    }


# ============================================================
# TOOL 6 — Top N
# ============================================================

def get_top_n(
    dataset,
    dimension: str,
    metric: str,
    n: int = 5
) -> dict[str, Any]:
    """
    Return the top N dimension values based on a metric.
    """

    grouped = group_by(
        dataset=dataset,
        dimension=dimension,
        metric=metric,
        aggregation="sum"
    )

    results = grouped["results"][:n]

    return {
        "operation": "top_n",
        "dataset": grouped["dataset"],
        "dimension": grouped["dimension"],
        "metric": grouped["metric"],
        "n": n,
        "results": results
    }


# ============================================================
# TOOL 7 — Time Trend
# ============================================================

def get_time_trend(
    dataset,
    date_column: str,
    metric: str,
    aggregation: str = "sum",
    grain: str = "month"
) -> dict[str, Any]:
    """
    Calculate a time-based trend.
    """

    df = _load_dataset(dataset)

    real_date_column = _find_column(
        df,
        date_column
    )

    real_metric = _find_column(
        df,
        metric
    )

    if real_date_column is None:
        raise ValueError(
            f"Date column '{date_column}' does not exist."
        )

    if real_metric is None:
        raise ValueError(
            f"Metric column '{metric}' does not exist."
        )

    df[real_date_column] = pd.to_datetime(
        df[real_date_column],
        errors="coerce"
    )

    df[real_metric] = pd.to_numeric(
        df[real_metric],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            real_date_column,
            real_metric
        ]
    )

    grain = grain.lower().strip()

    if grain == "day":
        df["_period"] = (
            df[real_date_column]
            .dt.to_period("D")
            .astype(str)
        )

    elif grain == "week":
        df["_period"] = (
            df[real_date_column]
            .dt.to_period("W")
            .astype(str)
        )

    elif grain == "month":
        df["_period"] = (
            df[real_date_column]
            .dt.to_period("M")
            .astype(str)
        )

    elif grain == "year":
        df["_period"] = (
            df[real_date_column]
            .dt.to_period("Y")
            .astype(str)
        )

    else:
        raise ValueError(
            "Grain must be day, week, month, or year."
        )

    if aggregation == "sum":
        grouped = df.groupby(
            "_period"
        )[real_metric].sum()

    elif aggregation == "mean":
        grouped = df.groupby(
            "_period"
        )[real_metric].mean()

    elif aggregation == "count":
        grouped = df.groupby(
            "_period"
        )[real_metric].count()

    else:
        raise ValueError(
            "Unsupported aggregation."
        )

    records = []

    for period, value in grouped.items():

        records.append({
            "period": str(period),
            "value": (
                float(value)
                if pd.notna(value)
                else None
            )
        })

    return {
        "operation": "time_trend",
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "date_column": str(real_date_column),
        "metric": str(real_metric),
        "aggregation": aggregation,
        "grain": grain,
        "results": records
    }


# ============================================================
# TOOL 8 — Dataset Summary
# ============================================================

def get_dataset_summary(
    dataset
) -> dict[str, Any]:
    """
    Return useful automatically detected metrics for a dataset.
    """

    df = _load_dataset(dataset)

    numeric_columns = [
        str(column)
        for column in df.select_dtypes(
            include="number"
        ).columns
    ]

    categorical_columns = [
        str(column)
        for column in df.select_dtypes(
            include=["object", "category"]
        ).columns
    ]

    date_column = _detect_date_column(df)

    revenue_column = _detect_numeric_column(
        df,
        [
            "revenue",
            "sales",
            "net",
            "amount",
            "income",
            "turnover"
        ]
    )

    quantity_column = _detect_numeric_column(
        df,
        [
            "quantity",
            "qty",
            "units",
            "count"
        ]
    )

    cost_column = _detect_numeric_column(
        df,
        [
            "cost",
            "expense"
        ]
    )

    dimension_column = _detect_dimension_column(df)

    return {
        "dataset": getattr(
            dataset,
            "original_filename",
            "unknown"
        ),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_column": (
            str(date_column)
            if date_column
            else None
        ),
        "detected_metrics": {
            "revenue": (
                str(revenue_column)
                if revenue_column
                else None
            ),
            "quantity": (
                str(quantity_column)
                if quantity_column
                else None
            ),
            "cost": (
                str(cost_column)
                if cost_column
                else None
            )
        },
        "recommended_dimension": (
            str(dimension_column)
            if dimension_column
            else None
        )
    }


def cross_dataset_analysis(
    primary_dataset,
    primary_dimension,
    primary_metric,
    lookup_dataset,
    lookup_key,
    return_column,
    aggregation="sum",
    ranking="highest",
    n=1,
):
    """
    Perform an analytical aggregation on one dataset and then
    look up the resulting dimension values in another dataset.

    Example:

    sample-raffi.xlsx
        site -> SUM(net)
        find highest site

    centre_data.csv
        site -> center

    Result:
        GZBINP -> corresponding center
    """

    primary_df = _load_dataset(primary_dataset)
    lookup_df = _load_dataset(lookup_dataset)

    # -----------------------------
    # Validate primary columns
    # -----------------------------

    if primary_dimension not in primary_df.columns:
        raise ValueError(
            f"Dimension column '{primary_dimension}' "
            f"not found in {primary_dataset.original_filename}"
        )

    if primary_metric not in primary_df.columns:
        raise ValueError(
            f"Metric column '{primary_metric}' "
            f"not found in {primary_dataset.original_filename}"
        )

    # -----------------------------
    # Validate lookup columns
    # -----------------------------

    if lookup_key not in lookup_df.columns:
        raise ValueError(
            f"Lookup column '{lookup_key}' "
            f"not found in {lookup_dataset.original_filename}"
        )

    if return_column not in lookup_df.columns:
        raise ValueError(
            f"Return column '{return_column}' "
            f"not found in {lookup_dataset.original_filename}"
        )

    # -----------------------------
    # Prepare metric
    # -----------------------------

    primary_df = primary_df.copy()

    primary_df[primary_metric] = pd.to_numeric(
        primary_df[primary_metric],
        errors="coerce"
    )

    primary_df = primary_df.dropna(
        subset=[primary_metric, primary_dimension]
    )

    if primary_df.empty:
        raise ValueError(
            "No valid records available for the requested analysis."
        )

    # -----------------------------
    # Aggregate
    # -----------------------------

    if aggregation == "sum":

        aggregated = (
            primary_df
            .groupby(primary_dimension, dropna=False)[primary_metric]
            .sum()
            .reset_index()
        )

    elif aggregation == "mean":

        aggregated = (
            primary_df
            .groupby(primary_dimension, dropna=False)[primary_metric]
            .mean()
            .reset_index()
        )

    elif aggregation == "count":

        aggregated = (
            primary_df
            .groupby(primary_dimension, dropna=False)[primary_metric]
            .count()
            .reset_index()
        )

    elif aggregation == "max":

        aggregated = (
            primary_df
            .groupby(primary_dimension, dropna=False)[primary_metric]
            .max()
            .reset_index()
        )

    elif aggregation == "min":

        aggregated = (
            primary_df
            .groupby(primary_dimension, dropna=False)[primary_metric]
            .min()
            .reset_index()
        )

    else:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    # -----------------------------
    # Ranking
    # -----------------------------

    ascending = ranking.lower() == "lowest"

    aggregated = aggregated.sort_values(
        by=primary_metric,
        ascending=ascending
    )

    aggregated = aggregated.head(n)

    # -----------------------------
    # Normalize lookup keys
    # -----------------------------

    aggregated["_lookup_key"] = (
        aggregated[primary_dimension]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    lookup_df = lookup_df.copy()

    lookup_df["_lookup_key"] = (
        lookup_df[lookup_key]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------
    # Remove duplicate lookup keys
    # -----------------------------

    lookup_unique = lookup_df[
        ["_lookup_key", return_column]
    ].drop_duplicates(
        subset=["_lookup_key"]
    )

    # -----------------------------
    # Perform lookup
    # -----------------------------

    result = aggregated.merge(
        lookup_unique,
        on="_lookup_key",
        how="left"
    )

    # -----------------------------
    # Build clean result
    # -----------------------------

    results = []

    for _, row in result.iterrows():

        lookup_value = row.get(return_column)

        if pd.isna(lookup_value):
            lookup_value = None

        results.append(
            {
                "dimension": row[primary_dimension],
                "metric_value": float(row[primary_metric]),
                "lookup_value": lookup_value,
            }
        )

    return {
        "success": True,
        "primary_dataset": primary_dataset.original_filename,
        "lookup_dataset": lookup_dataset.original_filename,
        "primary_dimension": primary_dimension,
        "primary_metric": primary_metric,
        "lookup_key": lookup_key,
        "return_column": return_column,
        "aggregation": aggregation,
        "ranking": ranking,
        "results": results,
    }