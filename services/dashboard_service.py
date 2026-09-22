import re
import pandas as pd

from services.data_service import DataService


class DashboardService:
    """
    Dynamic dashboard analytics engine.

    The service is designed to work with different business datasets
    without depending on one fixed schema.

    Python/Pandas performs all numerical calculations.
    The LLM should only interpret the validated results.
    """

    # ---------------------------------------------------------
    # COLUMN DETECTION KEYWORDS
    # ---------------------------------------------------------

    REVENUE_KEYWORDS = [
        "net_revenue",
        "net_sales",
        "sales_amount",
        "sale_amount",
        "revenue",
        "sales",
        "net",
        "gross",
        "amount",
    ]

    QUANTITY_KEYWORDS = [
        "quantity",
        "qty",
        "units",
        "unit_sold",
        "units_sold",
        "volume",
    ]

    COST_KEYWORDS = [
        "total_cost",
        "cost_value",
        "cost_amount",
        "cost",
        "expense",
        "expenses",
    ]

    DATE_KEYWORDS = [
        "date",
        "datetime",
        "timestamp",
        "transaction_date",
        "bill_date",
        "order_date",
        "created_at",
        "updated_at",
        "month",
        "year",
    ]

    DIMENSION_KEYWORDS = [
        "site",
        "center",
        "centre",
        "branch",
        "location",
        "region",
        "zone",
        "category",
        "department",
        "specialization",
        "speciality",
        "product",
        "item",
        "service",
        "type",
        "status",
    ]

    # Columns that are usually identifiers/PII rather than
    # meaningful analytical dimensions.
    EXCLUDED_DIMENSION_KEYWORDS = [
        "id",
        "email",
        "phone",
        "mobile",
        "address",
        "patient_name",
        "customer_name",
        "employee_name",
        "doctor_name",
        "bill_no",
        "invoice_no",
        "transaction_id",
        "trans_id",
        "mpid",
    ]

    # ---------------------------------------------------------
    # BASIC HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_column_name(column):
        """
        Convert a column name into a normalized form.

        Example:
            "Net Revenue" -> "net_revenue"
            "Net-Revenue" -> "net_revenue"
        """

        value = str(column).strip().lower()

        value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value
        )

        return value.strip("_")

    @staticmethod
    def _column_tokens(column):
        """
        Return useful tokens from a column name.
        """

        normalized = (
            DashboardService
            ._normalize_column_name(column)
        )

        return set(
            normalized.split("_")
        )

    # ---------------------------------------------------------
    # NUMERIC COLUMN DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def _find_best_numeric_column(
        dataframe,
        keywords,
        exclude=None
    ):
        """
        Find the best numeric column using semantic scoring.

        Exact semantic matches receive higher priority than
        generic partial matches.
        """

        exclude = exclude or set()

        candidates = []

        for column in dataframe.columns:

            if column in exclude:
                continue

            series = dataframe[column]

            if not pd.api.types.is_numeric_dtype(series):
                continue

            normalized = (
                DashboardService
                ._normalize_column_name(column)
            )

            tokens = (
                DashboardService
                ._column_tokens(column)
            )

            score = 0

            for index, keyword in enumerate(keywords):

                keyword_normalized = (
                    DashboardService
                    ._normalize_column_name(keyword)
                )

                keyword_tokens = set(
                    keyword_normalized.split("_")
                )

                # Exact column match
                if normalized == keyword_normalized:
                    score += 100 - index

                # Exact token match
                elif keyword_normalized in tokens:
                    score += 70 - index

                # Keyword contained in column
                elif keyword_normalized in normalized:
                    score += 40 - index

                # Token overlap
                elif tokens.intersection(keyword_tokens):
                    score += 25 - index

            # Ignore columns with no semantic match
            if score <= 0:
                continue

            candidates.append(
                (
                    score,
                    column
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1]

    # ---------------------------------------------------------
    # DATE DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def _find_date_column(dataframe):
        """
        Detect a usable date column.
        """

        # First check column names.
        named_candidates = []

        for column in dataframe.columns:

            normalized = (
                DashboardService
                ._normalize_column_name(column)
            )

            score = 0

            for keyword in DashboardService.DATE_KEYWORDS:

                keyword_normalized = (
                    DashboardService
                    ._normalize_column_name(keyword)
                )

                if normalized == keyword_normalized:
                    score += 100

                elif keyword_normalized in normalized:
                    score += 50

            if score > 0:
                named_candidates.append(
                    (
                        score,
                        column
                    )
                )

        named_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        for _, column in named_candidates:

            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce"
            )

            valid_ratio = (
                converted.notna().mean()
                if len(converted) > 0
                else 0
            )

            if valid_ratio >= 0.5:
                return column

        # Fallback: inspect object/string columns.
        for column in dataframe.select_dtypes(
            include=["object", "string"]
        ).columns:

            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce"
            )

            valid_ratio = (
                converted.notna().mean()
                if len(converted) > 0
                else 0
            )

            if valid_ratio >= 0.8:
                return column

        return None

    # ---------------------------------------------------------
    # DIMENSION DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def _is_excluded_dimension(column):
        """
        Determine whether a column should not be used as a
        business dimension.
        """

        normalized = (
            DashboardService
            ._normalize_column_name(column)
        )

        for excluded in (
            DashboardService
            .EXCLUDED_DIMENSION_KEYWORDS
        ):

            excluded_normalized = (
                DashboardService
                ._normalize_column_name(excluded)
            )

            if (
                normalized == excluded_normalized
                or excluded_normalized in normalized
            ):
                return True

        return False

    @staticmethod
    def _find_dimension_columns(
        dataframe,
        revenue_column=None,
        quantity_column=None,
        cost_column=None,
        date_column=None
    ):
        """
        Find useful categorical business dimensions.

        The method does not assume that 'site' exists.
        """

        excluded_columns = {
            revenue_column,
            quantity_column,
            cost_column,
            date_column,
        }

        dimensions = []

        for column in dataframe.columns:

            if column in excluded_columns:
                continue

            series = dataframe[column]

            # Dimensions should normally be categorical/string.
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
            ):
                continue

            if DashboardService._is_excluded_dimension(
                column
            ):
                continue

            unique_count = series.nunique(
                dropna=True
            )

            if unique_count < 2:
                continue

            # Avoid extremely high-cardinality text columns.
            if unique_count > 100:
                continue

            normalized = (
                DashboardService
                ._normalize_column_name(column)
            )

            tokens = (
                DashboardService
                ._column_tokens(column)
            )

            score = 0

            for index, keyword in enumerate(
                DashboardService.DIMENSION_KEYWORDS
            ):

                keyword_normalized = (
                    DashboardService
                    ._normalize_column_name(keyword)
                )

                if normalized == keyword_normalized:
                    score += 100 - index

                elif keyword_normalized in tokens:
                    score += 70 - index

                elif keyword_normalized in normalized:
                    score += 40 - index

            # Generic categorical fields are still potentially useful.
            if score == 0:
                score = 10

            # Prefer dimensions with reasonable cardinality.
            if unique_count <= 20:
                score += 10

            elif unique_count <= 50:
                score += 5

            dimensions.append(
                {
                    "column": column,
                    "score": score,
                    "unique_count": int(unique_count),
                }
            )

        dimensions.sort(
            key=lambda item: (
                item["score"],
                -item["unique_count"]
            ),
            reverse=True
        )

        return [
            item["column"]
            for item in dimensions
        ]

    # ---------------------------------------------------------
    # TIME PERFORMANCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_time_performance(
        dataframe,
        date_column,
        revenue_column
    ):
        """
        Generate a useful revenue time series.

        Short datasets:
            daily

        Longer datasets:
            monthly
        """

        if not date_column or not revenue_column:
            return {}

        temp = dataframe[
            [
                date_column,
                revenue_column
            ]
        ].copy()

        temp[date_column] = pd.to_datetime(
            temp[date_column],
            errors="coerce"
        )

        temp[revenue_column] = pd.to_numeric(
            temp[revenue_column],
            errors="coerce"
        )

        temp = temp.dropna()

        if temp.empty:
            return {}

        distinct_dates = (
            temp[date_column]
            .dt.normalize()
            .nunique()
        )

        if distinct_dates < 2:
            return {}

        date_range = (
            temp[date_column].max()
            - temp[date_column].min()
        ).days

        if date_range <= 60:

            temp["period"] = (
                temp[date_column]
                .dt.strftime("%d %b")
            )

        else:

            temp["period"] = (
                temp[date_column]
                .dt.to_period("M")
                .astype(str)
            )

        grouped = (
            temp
            .groupby("period")[revenue_column]
            .sum()
        )

        # Preserve chronological ordering.
        if date_range <= 60:

            order = (
                temp
                .groupby("period")[date_column]
                .min()
                .sort_values()
                .index
            )

            grouped = grouped.reindex(order)

        else:

            grouped = grouped.sort_index()

        return {
            str(key): round(
                float(value),
                2
            )
            for key, value in grouped.items()
        }

    # ---------------------------------------------------------
    # DIMENSION PERFORMANCE
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_dimension_performance(
        dataframe,
        dimension_column,
        revenue_column,
        limit=10
    ):
        """
        Calculate revenue grouped by a business dimension.
        """

        if (
            not dimension_column
            or not revenue_column
        ):
            return {}

        temp = dataframe[
            [
                dimension_column,
                revenue_column
            ]
        ].copy()

        temp[revenue_column] = pd.to_numeric(
            temp[revenue_column],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[
                dimension_column,
                revenue_column
            ]
        )

        if temp.empty:
            return {}

        grouped = (
            temp
            .groupby(
                dimension_column,
                dropna=False
            )[revenue_column]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(limit)
        )

        return {
            str(key): round(
                float(value),
                2
            )
            for key, value in grouped.items()
        }

    # ---------------------------------------------------------
    # QUANTITY BY DIMENSION
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_quantity_performance(
        dataframe,
        dimension_column,
        quantity_column,
        limit=10
    ):
        """
        Calculate quantity grouped by a business dimension.
        """

        if (
            not dimension_column
            or not quantity_column
        ):
            return {}

        temp = dataframe[
            [
                dimension_column,
                quantity_column
            ]
        ].copy()

        temp[quantity_column] = pd.to_numeric(
            temp[quantity_column],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=[
                dimension_column,
                quantity_column
            ]
        )

        if temp.empty:
            return {}

        grouped = (
            temp
            .groupby(
                dimension_column,
                dropna=False
            )[quantity_column]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(limit)
        )

        return {
            str(key): round(
                float(value),
                2
            )
            for key, value in grouped.items()
        }

    # ---------------------------------------------------------
    # PROFIT
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_profit(
        dataframe,
        revenue_column,
        cost_column
    ):
        """
        Calculate profit when both revenue and cost exist.
        """

        if (
            not revenue_column
            or not cost_column
        ):
            return None

        revenue = pd.to_numeric(
            dataframe[revenue_column],
            errors="coerce"
        )

        cost = pd.to_numeric(
            dataframe[cost_column],
            errors="coerce"
        )

        revenue = revenue.fillna(0)
        cost = cost.fillna(0)

        return round(
            float(
                revenue.sum()
                - cost.sum()
            ),
            2
        )

    # ---------------------------------------------------------
    # CATEGORY PERFORMANCE
    # ---------------------------------------------------------

    @staticmethod
    def _find_special_dimension(
        dimension_columns,
        keywords
    ):
        """
        Find a particular type of dimension.
        """

        for column in dimension_columns:

            normalized = (
                DashboardService
                ._normalize_column_name(column)
            )

            for keyword in keywords:

                if keyword in normalized:
                    return column

        return None

    @staticmethod
    def _calculate_category_performance(
        dataframe,
        dimension_columns,
        revenue_column
    ):

        category_column = (
            DashboardService
            ._find_special_dimension(
                dimension_columns,
                [
                    "category",
                    "group",
                    "department",
                    "type"
                ]
            )
        )

        if not category_column:
            return {}

        return (
            DashboardService
            ._calculate_dimension_performance(
                dataframe,
                category_column,
                revenue_column,
                limit=8
            )
        )

    @staticmethod
    def _calculate_product_performance(
        dataframe,
        dimension_columns,
        revenue_column
    ):

        product_column = (
            DashboardService
            ._find_special_dimension(
                dimension_columns,
                [
                    "product",
                    "item",
                    "service"
                ]
            )
        )

        if not product_column:
            return {}

        return (
            DashboardService
            ._calculate_dimension_performance(
                dataframe,
                product_column,
                revenue_column,
                limit=8
            )
        )

    # ---------------------------------------------------------
    # SINGLE DATASET ANALYSIS
    # ---------------------------------------------------------

    @staticmethod
    def analyze_dataset(dataset):
        """
        Analyse one dataset dynamically.
        """

        try:

            dataframe = DataService.load_dataset(
                dataset.file_path
            )

            if dataframe.empty:

                return {
                    "dataset_id": dataset.id,
                    "dataset_name": dataset.source_name,
                    "error": "Dataset is empty."
                }

            # -------------------------------------------------
            # DETECT IMPORTANT COLUMNS
            # -------------------------------------------------

            revenue_column = (
                DashboardService
                ._find_best_numeric_column(
                    dataframe,
                    DashboardService.REVENUE_KEYWORDS
                )
            )

            quantity_column = (
                DashboardService
                ._find_best_numeric_column(
                    dataframe,
                    DashboardService.QUANTITY_KEYWORDS
                )
            )

            cost_column = (
                DashboardService
                ._find_best_numeric_column(
                    dataframe,
                    DashboardService.COST_KEYWORDS
                )
            )

            date_column = (
                DashboardService
                ._find_date_column(
                    dataframe
                )
            )

            # -------------------------------------------------
            # DIMENSIONS
            # -------------------------------------------------

            dimension_columns = (
                DashboardService
                ._find_dimension_columns(
                    dataframe,
                    revenue_column,
                    quantity_column,
                    cost_column,
                    date_column
                )
            )

            # Primary dimension is the strongest candidate.
            primary_dimension = (
                dimension_columns[0]
                if dimension_columns
                else None
            )

            # -------------------------------------------------
            # REVENUE
            # -------------------------------------------------

            total_revenue = None
            average_transaction = None

            if revenue_column:

                revenue_values = pd.to_numeric(
                    dataframe[revenue_column],
                    errors="coerce"
                )

                valid_revenue = (
                    revenue_values.dropna()
                )

                if not valid_revenue.empty:

                    total_revenue = round(
                        float(
                            valid_revenue.sum()
                        ),
                        2
                    )

                    average_transaction = round(
                        float(
                            valid_revenue.mean()
                        ),
                        2
                    )

            # -------------------------------------------------
            # QUANTITY
            # -------------------------------------------------

            total_quantity = None

            if quantity_column:

                quantity_values = pd.to_numeric(
                    dataframe[quantity_column],
                    errors="coerce"
                )

                total_quantity = round(
                    float(
                        quantity_values
                        .fillna(0)
                        .sum()
                    ),
                    2
                )

            # -------------------------------------------------
            # PROFIT
            # -------------------------------------------------

            profit = (
                DashboardService
                ._calculate_profit(
                    dataframe,
                    revenue_column,
                    cost_column
                )
            )

            # -------------------------------------------------
            # PRIMARY DIMENSION
            # -------------------------------------------------

            dimension_performance = (
                DashboardService
                ._calculate_dimension_performance(
                    dataframe,
                    primary_dimension,
                    revenue_column
                )
            )

            # -------------------------------------------------
            # CATEGORY
            # -------------------------------------------------

            category_performance = (
                DashboardService
                ._calculate_category_performance(
                    dataframe,
                    dimension_columns,
                    revenue_column
                )
            )

            # -------------------------------------------------
            # PRODUCT
            # -------------------------------------------------

            product_performance = (
                DashboardService
                ._calculate_product_performance(
                    dataframe,
                    dimension_columns,
                    revenue_column
                )
            )

            # -------------------------------------------------
            # QUANTITY BY PRIMARY DIMENSION
            # -------------------------------------------------

            quantity_performance = (
                DashboardService
                ._calculate_quantity_performance(
                    dataframe,
                    primary_dimension,
                    quantity_column
                )
            )

            # -------------------------------------------------
            # TIME
            # -------------------------------------------------

            time_performance = (
                DashboardService
                ._calculate_time_performance(
                    dataframe,
                    date_column,
                    revenue_column
                )
            )

            # -------------------------------------------------
            # FINAL RESULT
            # -------------------------------------------------

            return {

                "dataset_id":
                    dataset.id,

                "dataset_name":
                    dataset.source_name,

                "rows":
                    int(dataframe.shape[0]),

                "columns":
                    int(dataframe.shape[1]),

                # Detected schema
                "revenue_column":
                    revenue_column,

                "quantity_column":
                    quantity_column,

                "cost_column":
                    cost_column,

                "date_column":
                    date_column,

                # KPIs
                "total_revenue":
                    total_revenue,

                "total_transactions":
                    int(dataframe.shape[0]),

                "average_transaction":
                    average_transaction,

                "total_quantity":
                    total_quantity,

                "profit":
                    profit,

                # Dimensions
                "dimension_column":
                    primary_dimension,

                "available_dimensions":
                    dimension_columns,

                # Analysis
                "dimension_performance":
                    dimension_performance,

                "category_performance":
                    category_performance,

                "product_performance":
                    product_performance,

                "quantity_performance":
                    quantity_performance,

                "time_performance":
                    time_performance,
            }

        except Exception as error:

            return {

                "dataset_id":
                    dataset.id,

                "dataset_name":
                    dataset.source_name,

                "error":
                    str(error)
            }

    # ---------------------------------------------------------
    # MULTIPLE DATASETS
    # ---------------------------------------------------------

    @staticmethod
    def analyze_selected_datasets(
        selected_datasets
    ):
        """
        Analyse all selected datasets independently.

        IMPORTANT:
        This method does NOT automatically merge datasets.

        Dataset relationships should be handled separately
        by RelationshipService before a future join/query layer
        combines related datasets.
        """

        results = []

        for dataset in selected_datasets:

            result = (
                DashboardService
                .analyze_dataset(
                    dataset
                )
            )

            results.append(result)

        return results