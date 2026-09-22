import pandas as pd
from itertools import combinations

from services.data_service import DataService


class RelationshipService:

    # Columns that are usually useful as relationships
    KEY_NAME_HINTS = {
        "id",
        "code",
        "key",
        "number",
        "no",
        "name",
        "site",
        "center",
        "centre",
        "customer",
        "patient",
        "product",
        "doctor",
        "physician",
        "category",
        "city",
        "state",
        "country",
    }

    # Columns that normally should NOT be used as join keys
    EXCLUDED_NAME_HINTS = {
        "amount",
        "revenue",
        "gross",
        "net",
        "tax",
        "discount",
        "price",
        "cost",
        "quantity",
        "margin",
        "percentage",
        "date",
        "time",
        "mobile",
        "phone",
        "email",
    }

    @staticmethod
    def normalize_column_name(name):
        """
        Normalize column names so that small naming differences
        do not prevent relationship detection.
        """

        return (
            str(name)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

    @staticmethod
    def is_candidate_column(column_name, series):
        """
        Determine whether a column is a reasonable candidate
        for joining two datasets.
        """

        normalized = (
            str(column_name)
            .strip()
            .lower()
        )

        # Avoid obvious measure columns
        for excluded in RelationshipService.EXCLUDED_NAME_HINTS:

            if excluded in normalized:
                return False

        # Avoid completely numeric continuous columns
        if pd.api.types.is_numeric_dtype(series):

            unique_count = series.nunique(
                dropna=True
            )

            total_count = len(series)

            if total_count == 0:
                return False

            # ID-like numeric columns can still be useful
            if (
                "id" in normalized
                or "code" in normalized
                or "no" in normalized
            ):
                return True

            # Otherwise avoid high-cardinality numeric measures
            if unique_count > 0.5 * total_count:
                return False

        return True

    @staticmethod
    def calculate_overlap(series_a, series_b):
        """
        Calculate how many values are shared between two columns.
        """

        values_a = set(
            series_a.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        values_b = set(
            series_b.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if not values_a or not values_b:
            return 0, 0.0

        common_values = values_a.intersection(
            values_b
        )

        # Symmetric overlap:
        # common values / smaller distinct set
        denominator = min(
            len(values_a),
            len(values_b)
        )

        overlap = (
            len(common_values) / denominator
            if denominator
            else 0
        )

        return (
            len(common_values),
            overlap * 100
        )

    @staticmethod
    def calculate_confidence(
        column_a,
        column_b,
        overlap_percentage
    ):
        """
        Calculate relationship confidence using:
        - column-name similarity
        - value overlap
        """

        name_a = (
            RelationshipService
            .normalize_column_name(column_a)
        )

        name_b = (
            RelationshipService
            .normalize_column_name(column_b)
        )

        confidence = 0

        # Exact normalized column name
        if name_a == name_b:
            confidence += 50

        # Partial name similarity
        elif (
            name_a in name_b
            or name_b in name_a
        ):
            confidence += 30

        # Value overlap
        if overlap_percentage >= 90:
            confidence += 50

        elif overlap_percentage >= 70:
            confidence += 40

        elif overlap_percentage >= 50:
            confidence += 30

        elif overlap_percentage >= 30:
            confidence += 20

        elif overlap_percentage >= 10:
            confidence += 10

        return min(confidence, 100)

    @staticmethod
    def detect_relationships(datasets):
        """
        Detect possible relationships between selected datasets.

        datasets:
            List of Dataset database objects.
        """

        relationships = []

        # Need at least two datasets
        if len(datasets) < 2:
            return relationships

        loaded_datasets = []

        # Load all datasets
        for dataset in datasets:

            try:

                dataframe = DataService.load_dataset(
                    dataset.file_path
                )

                loaded_datasets.append(
                    {
                        "dataset": dataset,
                        "dataframe": dataframe,
                    }
                )

            except Exception:
                continue

        # Compare every dataset pair
        for left, right in combinations(
            loaded_datasets,
            2
        ):

            left_dataset = left["dataset"]
            right_dataset = right["dataset"]

            left_df = left["dataframe"]
            right_df = right["dataframe"]

            # Compare candidate columns
            for left_column in left_df.columns:

                if not RelationshipService.is_candidate_column(
                    left_column,
                    left_df[left_column]
                ):
                    continue

                for right_column in right_df.columns:

                    if not RelationshipService.is_candidate_column(
                        right_column,
                        right_df[right_column]
                    ):
                        continue

                    common_values, overlap = (
                        RelationshipService.calculate_overlap(
                            left_df[left_column],
                            right_df[right_column]
                        )
                    )

                    if common_values == 0:
                        continue

                    confidence = (
                        RelationshipService
                        .calculate_confidence(
                            left_column,
                            right_column,
                            overlap
                        )
                    )

                    # Only return reasonably strong relationships
                    if confidence >= 50:

                        relationships.append(
                            {
                                "left_dataset_id":
                                    left_dataset.id,

                                "left_dataset_name":
                                    left_dataset.source_name,

                                "left_column":
                                    left_column,

                                "right_dataset_id":
                                    right_dataset.id,

                                "right_dataset_name":
                                    right_dataset.source_name,

                                "right_column":
                                    right_column,

                                "common_values":
                                    common_values,

                                "overlap_percentage":
                                    round(
                                        overlap,
                                        2
                                    ),

                                "confidence":
                                    confidence,
                            }
                        )

        # Remove duplicate/weak relationships
        relationships.sort(
            key=lambda x: (
                x["confidence"],
                x["overlap_percentage"],
                x["common_values"]
            ),
            reverse=True
        )

        return relationships