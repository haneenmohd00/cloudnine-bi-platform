from pathlib import Path
import pandas as pd


class DataService:

    ALLOWED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    @staticmethod
    def is_allowed_file(filename):
        """Check whether the uploaded file type is supported."""

        extension = Path(filename).suffix.lower()

        return extension in DataService.ALLOWED_EXTENSIONS

    @staticmethod
    def load_dataset(file_path):
        """Load CSV or Excel datasets dynamically."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension == ".csv":

            dataframe = pd.read_csv(path)

        elif extension in [".xlsx", ".xls"]:

            dataframe = pd.read_excel(path)

        else:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return dataframe

    @staticmethod
    def get_dataset_metadata(dataframe):
        """Return generic metadata about a dataset."""

        metadata = {
            "rows": dataframe.shape[0],
            "columns_count": dataframe.shape[1],
            "columns": dataframe.columns.tolist(),
            "data_types": {
                column: str(dtype)
                for column, dtype in dataframe.dtypes.items()
            },
            "missing_values": dataframe.isnull().sum().to_dict()
        }

        return metadata

    # ==========================================================
    # MULTI-DATASET JOIN ENGINE
    # ==========================================================

    @staticmethod
    def normalize_column_name(column):
        """
        Normalize a column name only for comparison.

        Example:
            Site       -> site
            SITE       -> site
            site_name  -> site_name
        """

        return (
            str(column)
            .strip()
            .lower()
        )

    @staticmethod
    def find_common_columns(left_df, right_df):
        """
        Find columns shared by two datasets.

        Matching is case-insensitive and ignores
        surrounding whitespace.
        """

        left_columns = {
            DataService.normalize_column_name(column): column
            for column in left_df.columns
        }

        right_columns = {
            DataService.normalize_column_name(column): column
            for column in right_df.columns
        }

        common = []

        for normalized_name in left_columns:

            if normalized_name in right_columns:

                common.append(
                    (
                        left_columns[normalized_name],
                        right_columns[normalized_name]
                    )
                )

        return common

    @staticmethod
    def _is_unique_key(dataframe, column):
        """
        Check whether a column can safely act as
        a lookup/dimension key.
        """

        if column not in dataframe.columns:
            return False

        values = dataframe[column].dropna()

        if values.empty:
            return False

        return values.is_unique

    @staticmethod
    def _prepare_join_columns(left_df, right_df, left_column, right_column):
        """
        Make compatible copies of join columns.

        This handles common situations such as:
            int vs string
            whitespace
            different capitalization
        """

        left = left_df.copy()
        right = right_df.copy()

        left["_join_key"] = (
            left[left_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        right["_join_key"] = (
            right[right_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return left, right

    @staticmethod
    def build_combined_dataset(datasets):
        """
        Combine multiple selected datasets when safe
        relationships can be detected automatically.

        Returns:
            {
                "dataframe": combined dataframe,
                "joins": information about joins performed,
                "unjoined": datasets that could not be joined,
                "relationships": detected relationships
            }
        """

        if not datasets:
            return {
                "dataframe": pd.DataFrame(),
                "joins": [],
                "unjoined": [],
                "relationships": []
            }

        # ------------------------------------------------------
        # Load all datasets
        # ------------------------------------------------------

        loaded = []

        for dataset in datasets:

            dataframe = DataService.load_dataset(
                dataset.file_path
            )

            loaded.append(
                {
                    "dataset": dataset,
                    "dataframe": dataframe
                }
            )

        # ------------------------------------------------------
        # Single dataset
        # ------------------------------------------------------

        if len(loaded) == 1:

            return {
                "dataframe": loaded[0]["dataframe"].copy(),
                "joins": [],
                "unjoined": [],
                "relationships": []
            }

        # ------------------------------------------------------
        # Start with the largest dataset.
        #
        # Usually the transaction/fact table is larger,
        # while lookup/dimension tables are smaller.
        # ------------------------------------------------------

        loaded.sort(
            key=lambda item: len(item["dataframe"]),
            reverse=True
        )

        combined = loaded[0]["dataframe"].copy()

        base_dataset = loaded[0]["dataset"]

        remaining = loaded[1:]

        joins = []
        unjoined = []
        relationships = []

        # ------------------------------------------------------
        # Join remaining datasets one by one
        # ------------------------------------------------------

        for item in remaining:

            other_dataset = item["dataset"]

            other_df = item["dataframe"].copy()

            common_columns = DataService.find_common_columns(
                combined,
                other_df
            )

            if not common_columns:

                unjoined.append(
                    {
                        "dataset": other_dataset.source_name,
                        "reason": "No common columns detected."
                    }
                )

                continue

            # --------------------------------------------------
            # Find a safe relationship.
            #
            # Prefer a key that is unique in the smaller
            # lookup/dimension dataset.
            # --------------------------------------------------

            selected_relationship = None

            for left_column, right_column in common_columns:

                right_unique = DataService._is_unique_key(
                    other_df,
                    right_column
                )

                left_unique = DataService._is_unique_key(
                    combined,
                    left_column
                )

                # Best case:
                # combined = many
                # other = one
                if right_unique:

                    selected_relationship = (
                        left_column,
                        right_column,
                        "many_to_one"
                    )

                    break

                # One-to-one
                if left_unique and right_unique:

                    selected_relationship = (
                        left_column,
                        right_column,
                        "one_to_one"
                    )

                    break

            # --------------------------------------------------
            # No safe relationship
            # --------------------------------------------------

            if selected_relationship is None:

                unjoined.append(
                    {
                        "dataset": other_dataset.source_name,
                        "reason": (
                            "Common columns were found, "
                            "but no safe unique lookup key "
                            "was detected."
                        ),
                        "common_columns": [
                            left
                            for left, right in common_columns
                        ]
                    }
                )

                continue

            left_column = selected_relationship[0]
            right_column = selected_relationship[1]
            relationship_type = selected_relationship[2]

            # --------------------------------------------------
            # Prepare temporary join keys
            # --------------------------------------------------

            left_prepared, right_prepared = (
                DataService._prepare_join_columns(
                    combined,
                    other_df,
                    left_column,
                    right_column
                )
            )

            # --------------------------------------------------
            # Avoid duplicate column names.
            #
            # Keep the join column from the combined dataframe.
            # --------------------------------------------------

            columns_to_add = []

            for column in right_prepared.columns:

                if column == "_join_key":
                    continue

                if column == right_column:
                    continue

                if column not in left_prepared.columns:

                    columns_to_add.append(column)

            right_for_merge = right_prepared[
                ["_join_key"] + columns_to_add
            ]

            # --------------------------------------------------
            # Perform LEFT JOIN
            # --------------------------------------------------

            combined = left_prepared.merge(
                right_for_merge,
                on="_join_key",
                how="left",
                validate="many_to_one"
            )

            combined.drop(
                columns=["_join_key"],
                inplace=True
            )

            # --------------------------------------------------
            # Record relationship
            # --------------------------------------------------

            join_info = {
                "left_dataset": base_dataset.source_name,
                "right_dataset": other_dataset.source_name,
                "left_column": left_column,
                "right_column": right_column,
                "relationship": relationship_type
            }

            joins.append(join_info)

            relationships.append(
                join_info
            )

        return {
            "dataframe": combined,
            "joins": joins,
            "unjoined": unjoined,
            "relationships": relationships
        }