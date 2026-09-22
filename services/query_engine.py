from __future__ import annotations

from typing import Any

import pandas as pd

from services.bi_tools import _load_dataset


class QueryEngine:
    """
    Generic SQL-like query engine for BI reports.

    The LLM does NOT execute data operations directly.

    Groq produces a structured query specification.
    QueryEngine validates and executes that specification
    using Pandas.
    """

    SUPPORTED_OPERATORS = {
        "=",
        "==",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
        "contains",
        "not_contains",
        "starts_with",
        "ends_with",
        "in",
        "not_in",
        "is_null",
        "is_not_null",
        "between",
    }

    SUPPORTED_AGGREGATIONS = {
        "sum",
        "mean",
        "average",
        "min",
        "max",
        "count",
        "nunique",
    }

    # ==========================================================
    # PUBLIC ENTRY POINT
    # ==========================================================

    @classmethod
    def execute(cls, dataset, query: dict[str, Any]) -> pd.DataFrame:
        """
        Execute a structured query against one dataset.

        Example:

        {
            "columns": ["patient_name", "site", "net"],

            "filters": [
                {
                    "column": "site",
                    "operator": "=",
                    "value": "GZBINP"
                }
            ],

            "group_by": [],

            "aggregations": [],

            "order_by": [],

            "limit": 100
        }
        """

        if not isinstance(query, dict):
            raise ValueError(
                "Query must be a dictionary."
            )

        dataframe = _load_dataset(dataset)

        dataframe = dataframe.copy()

        # ------------------------------------------------------
        # 1. Filters
        # ------------------------------------------------------

        dataframe = cls._apply_filters(
            dataframe,
            query.get("filters", [])
        )

        # ------------------------------------------------------
        # 2. Grouping / aggregation
        # ------------------------------------------------------

        group_by = query.get(
            "group_by",
            []
        )

        aggregations = query.get(
            "aggregations",
            []
        )

        if group_by or aggregations:

            dataframe = cls._apply_grouping(
                dataframe=dataframe,
                group_by=group_by,
                aggregations=aggregations
            )

        # ------------------------------------------------------
        # 3. Select columns
        # ------------------------------------------------------

        columns = query.get(
            "columns",
            ["*"]
        )

        dataframe = cls._select_columns(
            dataframe,
            columns
        )

        # ------------------------------------------------------
        # 4. Ordering
        # ------------------------------------------------------

        dataframe = cls._apply_ordering(
            dataframe,
            query.get("order_by", [])
        )

        # ------------------------------------------------------
        # 5. Limit
        # ------------------------------------------------------

        limit = query.get("limit")

        if limit is not None:

            try:
                limit = int(limit)
            except (TypeError, ValueError):

                raise ValueError(
                    "Limit must be an integer."
                )

            if limit < 1:

                raise ValueError(
                    "Limit must be greater than zero."
                )

            dataframe = dataframe.head(limit)

        # ------------------------------------------------------
        # 6. Clean index
        # ------------------------------------------------------

        dataframe = dataframe.reset_index(drop=True)

        return dataframe


    @classmethod
    def execute_multi_dataset(cls,datasets,query):
        """
        Execute a query across multiple datasets.

        The query must specify:

        base_dataset_id

        joins:
        [
            {
                "dataset_id": 2,
                "left_column": "site",
                "right_column": "site",
                "how": "left"
            }
        ]

        Example:

        {
            "base_dataset_id": 3,

            "joins": [
                {
                    "dataset_id": 2,
                    "left_column": "site",
                    "right_column": "site",
                    "how": "left"
                }
            ],

            "columns": [
                "site",
                "net",
                "center",
                "regional_director"
            ],

            "filters": [],

            "group_by": [],

            "aggregations": [],

            "order_by": [],

            "limit": 10
        }
        """

        if not datasets:

            raise ValueError(
                "At least one dataset is required."
            )

        if not isinstance(query, dict):

            raise ValueError(
                "Query must be a dictionary."
            )

        # =====================================================
        # Build dataset lookup
        # =====================================================

        dataset_map = {
            int(dataset.id): dataset
            for dataset in datasets
        }

        base_dataset_id = query.get(
            "base_dataset_id"
        )

        if base_dataset_id is None:

            # Backwards-compatible fallback
            base_dataset_id = datasets[0].id

        try:

            base_dataset_id = int(
                base_dataset_id
            )

        except (TypeError, ValueError):

            raise ValueError(
                "base_dataset_id must be an integer."
            )

        if base_dataset_id not in dataset_map:

            raise ValueError(
                f"Base dataset {base_dataset_id} "
                "is not available."
            )

        # =====================================================
        # Load base dataframe
        # =====================================================

        base_dataset = dataset_map[
            base_dataset_id
        ]

        dataframe = _load_dataset(
            base_dataset
        ).copy()

        # =====================================================
        # Perform joins
        # =====================================================

        joins = query.get(
            "joins",
            []
        )

        joined_dataset_ids = {
            base_dataset_id
        }

        for join in joins:

            if not isinstance(join, dict):

                raise ValueError(
                    "Each join must be an object."
                )

            target_dataset_id = join.get(
                "dataset_id"
            )

            left_column = join.get(
                "left_column"
            )

            right_column = join.get(
                "right_column"
            )

            join_type = join.get(
                "how",
                "left"
            ).lower()

            if target_dataset_id is None:

                raise ValueError(
                    "Join dataset_id is required."
                )

            try:

                target_dataset_id = int(
                    target_dataset_id
                )

            except (TypeError, ValueError):

                raise ValueError(
                    "Join dataset_id must be an integer."
                )

            if target_dataset_id not in dataset_map:

                raise ValueError(
                    f"Join dataset {target_dataset_id} "
                    "is not available."
                )

            if target_dataset_id in joined_dataset_ids:

                raise ValueError(
                    f"Dataset {target_dataset_id} "
                    "has already been joined."
                )

            if not left_column or not right_column:

                raise ValueError(
                    "Both left_column and right_column "
                    "are required for a join."
                )

            if join_type not in {
                "left",
                "inner",
                "right",
                "outer"
            }:

                raise ValueError(
                    f"Unsupported join type: {join_type}"
                )

            target_dataset = dataset_map[
                target_dataset_id
            ]

            target_dataframe = _load_dataset(
                target_dataset
            ).copy()

            # ---------------------------------------------
            # Validate join columns
            # ---------------------------------------------

            if left_column not in dataframe.columns:

                raise ValueError(
                    f"Join column '{left_column}' "
                    "does not exist in the current dataset."
                )

            if right_column not in target_dataframe.columns:

                raise ValueError(
                    f"Join column '{right_column}' "
                    f"does not exist in dataset "
                    f"{target_dataset_id}."
                )

            # ---------------------------------------------
            # Avoid duplicate non-key column collisions
            # ---------------------------------------------

            overlapping_columns = (
                set(dataframe.columns)
                & set(target_dataframe.columns)
            )

            overlapping_columns.discard(
                left_column
            )

            if right_column in overlapping_columns:

                overlapping_columns.discard(
                    right_column
                )

            rename_map = {}

            for column in overlapping_columns:

                rename_map[column] = (
                    f"{column}_"
                    f"dataset_{target_dataset_id}"
                )

            if rename_map:

                target_dataframe = (
                    target_dataframe
                    .rename(
                        columns=rename_map
                    )
                )

            # ---------------------------------------------
            # Perform merge
            # ---------------------------------------------

            dataframe = dataframe.merge(
                target_dataframe,
                how=join_type,
                left_on=left_column,
                right_on=right_column,
                suffixes=(
                    "",
                    f"_dataset_{target_dataset_id}"
                )
            )

            joined_dataset_ids.add(
                target_dataset_id
            )

        # =====================================================
        # Apply filters
        # =====================================================

        dataframe = cls._apply_filters(
            dataframe,
            query.get(
                "filters",
                []
            )
        )

        # =====================================================
        # Grouping / aggregation
        # =====================================================

        group_by = query.get(
            "group_by",
            []
        )

        aggregations = query.get(
            "aggregations",
            []
        )

        if group_by or aggregations:

            dataframe = cls._apply_grouping(
                dataframe=dataframe,
                group_by=group_by,
                aggregations=aggregations
            )

        # =====================================================
        # Select columns
        # =====================================================

        columns = query.get(
            "columns",
            ["*"]
        )

        dataframe = cls._select_columns(
            dataframe,
            columns
        )

        # =====================================================
        # Ordering
        # =====================================================

        dataframe = cls._apply_ordering(
            dataframe,
            query.get(
                "order_by",
                []
            )
        )

        # =====================================================
        # Limit
        # =====================================================

        limit = query.get(
            "limit"
        )

        if limit is not None:

            try:

                limit = int(limit)

            except (TypeError, ValueError):

                raise ValueError(
                    "Limit must be an integer."
                )

            if limit < 1:

                raise ValueError(
                    "Limit must be greater than zero."
                )

            dataframe = dataframe.head(
                limit
            )

        return dataframe.reset_index(
            drop=True
        )
    
    # ==========================================================
    # COLUMN VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame,columns: list[str]):
        """
        Make sure requested columns exist.
        """

        if not columns:
            return

        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Column(s) not found: "
                + ", ".join(missing)
            )

    # ==========================================================
    # FILTERS
    # ==========================================================

    @classmethod
    def _apply_filters(
        cls,
        dataframe: pd.DataFrame,
        filters: list[dict[str, Any]]) -> pd.DataFrame:

        if not filters:
            return dataframe

        result = dataframe.copy()

        for condition in filters:

            if not isinstance(condition, dict):

                raise ValueError(
                    "Each filter must be an object."
                )

            column = condition.get("column")
            operator = condition.get("operator")
            value = condition.get("value")

            if not column:

                raise ValueError(
                    "Filter column is required."
                )

            if not operator:

                raise ValueError(
                    f"Filter operator is required for '{column}'."
                )

            operator = operator.lower()

            if operator not in cls.SUPPORTED_OPERATORS:

                raise ValueError(
                    f"Unsupported filter operator: {operator}"
                )

            cls._validate_columns(
                result,
                [column]
            )

            series = result[column]

            # ----------------------------------------------
            # Null checks
            # ----------------------------------------------

            if operator == "is_null":

                mask = series.isna()

            elif operator == "is_not_null":

                mask = series.notna()

            # ----------------------------------------------
            # Equality
            # ----------------------------------------------

            elif operator in {"=", "=="}:

                mask = cls._comparison_mask(
                    series,
                    value,
                    "eq"
                )

            elif operator == "!=":

                mask = cls._comparison_mask(
                    series,
                    value,
                    "ne"
                )

            # ----------------------------------------------
            # Numeric comparisons
            # ----------------------------------------------

            elif operator in {
                ">",
                ">=",
                "<",
                "<="
            }:

                mask = cls._numeric_comparison(
                    series,
                    value,
                    operator
                )

            # ----------------------------------------------
            # Text operations
            # ----------------------------------------------

            elif operator == "contains":

                mask = (
                    series
                    .astype(str)
                    .str.contains(
                        str(value),
                        case=False,
                        na=False
                    )
                )

            elif operator == "not_contains":

                mask = ~(
                    series
                    .astype(str)
                    .str.contains(
                        str(value),
                        case=False,
                        na=False
                    )
                )

            elif operator == "starts_with":

                mask = (
                    series
                    .astype(str)
                    .str.lower()
                    .str.startswith(
                        str(value).lower()
                    )
                )

            elif operator == "ends_with":

                mask = (
                    series
                    .astype(str)
                    .str.lower()
                    .str.endswith(
                        str(value).lower()
                    )
                )

            # ----------------------------------------------
            # IN
            # ----------------------------------------------

            elif operator == "in":

                if not isinstance(value, list):

                    raise ValueError(
                        f"'in' filter for '{column}' "
                        "requires a list."
                    )

                mask = cls._case_insensitive_isin(
                    series,
                    value
                )

            elif operator == "not_in":

                if not isinstance(value, list):

                    raise ValueError(
                        f"'not_in' filter for '{column}' "
                        "requires a list."
                    )

                mask = ~cls._case_insensitive_isin(
                    series,
                    value
                )

            # ----------------------------------------------
            # BETWEEN
            # ----------------------------------------------

            elif operator == "between":

                if (
                    not isinstance(value, list)
                    or len(value) != 2
                ):

                    raise ValueError(
                        f"'between' filter for '{column}' "
                        "requires [minimum, maximum]."
                    )

                numeric_series = pd.to_numeric(
                    series,
                    errors="coerce"
                )

                lower = float(value[0])
                upper = float(value[1])

                mask = (
                    numeric_series >= lower
                ) & (
                    numeric_series <= upper
                )

            else:

                raise ValueError(
                    f"Unsupported operator: {operator}"
                )

            result = result.loc[mask]

        return result

    # ==========================================================
    # COMPARISON HELPERS
    # ==========================================================

    @staticmethod
    def _comparison_mask(
        series: pd.Series,
        value: Any,
        comparison: str
    ) -> pd.Series:
        """
        Compare values intelligently.

        Numeric columns are compared numerically.
        Text columns are compared case-insensitively.
        """

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        try:

            numeric_value = float(value)

            if numeric_series.notna().any():

                if comparison == "eq":
                    return numeric_series == numeric_value

                return numeric_series != numeric_value

        except (TypeError, ValueError):
            pass

        left = series.astype(str).str.strip().str.lower()
        right = str(value).strip().lower()

        if comparison == "eq":
            return left == right

        return left != right

    @staticmethod
    def _numeric_comparison(
        series: pd.Series,
        value: Any,
        operator: str
    ) -> pd.Series:

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):

            raise ValueError(
                f"Value '{value}' must be numeric "
                f"for operator '{operator}'."
            )

        if operator == ">":
            return numeric_series > numeric_value

        if operator == ">=":
            return numeric_series >= numeric_value

        if operator == "<":
            return numeric_series < numeric_value

        if operator == "<=":
            return numeric_series <= numeric_value

        raise ValueError(
            f"Unsupported numeric operator: {operator}"
        )

    @staticmethod
    def _case_insensitive_isin(
        series: pd.Series,
        values: list[Any]
    ) -> pd.Series:

        normalized_series = (
            series
            .astype(str)
            .str.strip()
            .str.lower()
        )

        normalized_values = {
            str(value).strip().lower()
            for value in values
        }

        return normalized_series.isin(
            normalized_values
        )

    # ==========================================================
    # GROUPING / AGGREGATION
    # ==========================================================

    @classmethod
    def _apply_grouping(
        cls,
        dataframe: pd.DataFrame,
        group_by: list[str],
        aggregations: list[dict[str, Any]]
    ) -> pd.DataFrame:

        cls._validate_columns(
            dataframe,
            group_by
        )

        if not aggregations:

            if not group_by:

                raise ValueError(
                    "Grouping requires at least one aggregation."
                )

            return (
                dataframe
                .groupby(
                    group_by,
                    dropna=False
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

        aggregation_map = {}

        rename_map = {}

        for aggregation in aggregations:

            if not isinstance(aggregation, dict):

                raise ValueError(
                    "Each aggregation must be an object."
                )

            column = aggregation.get(
                "column"
            )

            function = aggregation.get(
                "aggregation",
                "sum"
            ).lower()

            alias = aggregation.get(
                "alias"
            )

            if not column:

                raise ValueError(
                    "Aggregation column is required."
                )

            cls._validate_columns(
                dataframe,
                [column]
            )

            if function not in cls.SUPPORTED_AGGREGATIONS:

                raise ValueError(
                    f"Unsupported aggregation: {function}"
                )

            if function == "average":
                function = "mean"

            if function == "count":

                aggregation_map[column] = "count"

            elif function == "nunique":

                aggregation_map[column] = "nunique"

            else:

                numeric_series = pd.to_numeric(
                    dataframe[column],
                    errors="coerce"
                )

                dataframe = dataframe.copy()

                dataframe[column] = numeric_series

                aggregation_map[column] = function

            if alias:

                rename_map[column] = alias

        if not group_by:

            result = {}

            for column, function in aggregation_map.items():

                if function == "count":
                    value = dataframe[column].count()

                elif function == "nunique":
                    value = dataframe[column].nunique()

                else:
                    value = getattr(
                        dataframe[column],
                        function
                    )()

                alias = rename_map.get(
                    column,
                    column
                )

                result[alias] = [value]

            return pd.DataFrame(result)

        result = (
            dataframe
            .groupby(
                group_by,
                dropna=False
            )
            .agg(aggregation_map)
            .reset_index()
        )

        # Pandas can produce MultiIndex columns when multiple
        # aggregations are requested.
        if isinstance(
            result.columns,
            pd.MultiIndex
        ):

            flattened_columns = []

            for column in result.columns:

                if isinstance(column, tuple):

                    base = column[0]
                    function = column[1]

                    if base in rename_map:

                        flattened_columns.append(
                            rename_map[base]
                        )

                    else:

                        flattened_columns.append(
                            f"{base}_{function}"
                        )

                else:

                    flattened_columns.append(
                        str(column)
                    )

            result.columns = flattened_columns

        else:

            result = result.rename(
                columns=rename_map
            )

        return result

    # ==========================================================
    # COLUMN SELECTION
    # ==========================================================

    @classmethod
    def _select_columns(
        cls,
        dataframe: pd.DataFrame,
        columns: list[str]
    ) -> pd.DataFrame:

        if not columns:

            return dataframe

        if "*" in columns:

            return dataframe

        cls._validate_columns(
            dataframe,
            columns
        )

        return dataframe[
            columns
        ]

    # ==========================================================
    # ORDER BY
    # ==========================================================

    @classmethod
    def _apply_ordering(
        cls,
        dataframe: pd.DataFrame,
        order_by: list[dict[str, Any]]
    ) -> pd.DataFrame:

        if not order_by:

            return dataframe

        columns = []
        ascending = []

        for order in order_by:

            column = order.get(
                "column"
            )

            direction = order.get(
                "direction",
                "asc"
            ).lower()

            if not column:

                raise ValueError(
                    "Order-by column is required."
                )

            if direction not in {
                "asc",
                "desc"
            }:

                raise ValueError(
                    "Order direction must be 'asc' or 'desc'."
                )

            cls._validate_columns(
                dataframe,
                [column]
            )

            columns.append(column)
            ascending.append(
                direction == "asc"
            )

        return dataframe.sort_values(
            by=columns,
            ascending=ascending
        )