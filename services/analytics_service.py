import pandas as pd
import re

from services.data_service import DataService


def load_dataset(dataset):
    """
    Load a dataset using the existing DataService.
    """

    return DataService.load_dataset(
        dataset.file_path
    )


def normalize_text(text):
    """
    Normalize text for column matching.
    """

    return (
        str(text)
        .lower()
        .replace("_", " ")
        .strip()
    )


def find_column(question, columns):
    """
    Find the numeric/metric column mentioned
    in the user's question.
    """

    question = normalize_text(question)

    question_without_grouping = question

    grouping_patterns = [
        r"\bby\s+[a-zA-Z_ ]+",
        r"\bper\s+[a-zA-Z_ ]+",
        r"\bfor each\s+[a-zA-Z_ ]+"
    ]

    for pattern in grouping_patterns:

        question_without_grouping = re.sub(
            pattern,
            "",
            question_without_grouping
        ).strip()

    # First: look for an actual column name
    for column in columns:

        normalized_column = normalize_text(
            column
        )

        if normalized_column in question_without_grouping:

            return column

    # Business aliases
    aliases = {
        "revenue": [
            "net",
            "gross",
            "net_value",
            "gross_value"
        ],
        "sales": [
            "net",
            "gross",
            "sales",
            "revenue"
        ],
        "profit": [
            "profit",
            "margin",
            "net_profit"
        ]
    }

    for keyword, possible_columns in aliases.items():

        if keyword in question_without_grouping:

            for possible_column in possible_columns:

                if possible_column in columns:

                    return possible_column

    return None


def find_group_by_column(question, columns):
    """
    Detect the column used for GROUP BY.
    """

    question = normalize_text(question)

    patterns = [
        r"by ([a-zA-Z_ ]+)",
        r"per ([a-zA-Z_ ]+)",
        r"for each ([a-zA-Z_ ]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question
        )

        if match:

            possible_name = (
                match.group(1)
                .strip()
            )

            for column in columns:

                if normalize_text(column) == possible_name:

                    return column

    return None


def detect_operation(question):
    """
    Detect the requested analytical operation.
    """

    question = question.lower()

    if any(
        word in question
        for word in [
            "average",
            "avg",
            "mean"
        ]
    ):
        return "average"

    if any(
        word in question
        for word in [
            "total",
            "sum"
        ]
    ):
        return "sum"

    if any(
        word in question
        for word in [
            "count",
            "how many",
            "number of"
        ]
    ):
        return "count"

    return None


def analyze_question(
    question,
    datasets=None,
    dataframe=None
):

    operation = detect_operation(question)

    if not operation:
        return None

    try:

        # If the Flask route already prepared
        # a combined dataframe, use it directly.
        if dataframe is not None:

            analysis_dataframe = dataframe

            if datasets:
                dataset_name = " + ".join(
                    dataset.source_name
                    for dataset in datasets
                )
            else:
                dataset_name = "Selected dataset(s)"

        # Otherwise build the dataframe here.
        else:

            if not datasets:
                return None

            if len(datasets) == 1:

                analysis_dataframe = load_dataset(
                    datasets[0]
                )

                dataset_name = (
                    datasets[0].source_name
                )

            else:

                combined_result = (
                    DataService.build_combined_dataset(
                        datasets
                    )
                )

                analysis_dataframe = (
                    combined_result["dataframe"]
                )

                dataset_name = " + ".join(
                    dataset.source_name
                    for dataset in datasets
                )

        # From here onward, use the dataframe
        # that was actually selected/prepared.
        columns = list(
            analysis_dataframe.columns
        )

        target_column = find_column(
            question,
            columns
        )

        group_column = find_group_by_column(
            question,
            columns
        )

        # COUNT
        if operation == "count":

            if group_column:

                result = (
                    analysis_dataframe
                    .groupby(group_column)
                    .size()
                    .reset_index(name="count")
                )

                return format_result(
                    result,
                    dataset_name,
                    "Count"
                )

            total_count = len(
                analysis_dataframe
            )

            return (
                f"Datasets: {dataset_name}\n\n"
                f"Total number of records: "
                f"{total_count}"
            )

        # SUM / AVERAGE
        if not target_column:

            return (
                "I could not identify the "
                "column to analyze from the "
                "selected dataset(s)."
            )

        if not pd.api.types.is_numeric_dtype(
            analysis_dataframe[target_column]
        ):

            return (
                f"The column '{target_column}' "
                f"is not numeric, so I cannot "
                f"perform a {operation} calculation."
            )

        # SUM
        if operation == "sum":

            if group_column:

                result = (
                    analysis_dataframe
                    .groupby(group_column)[target_column]
                    .sum()
                    .reset_index()
                )

                result = result.rename(
                    columns={
                        target_column:
                        f"total_{target_column}"
                    }
                )

                return format_result(
                    result,
                    dataset_name,
                    f"Total {target_column}"
                )

            result = analysis_dataframe[
                target_column
            ].sum()

            return (
                f"Datasets: {dataset_name}\n\n"
                f"Total {target_column}: "
                f"{result:,.2f}"
            )

        # AVERAGE
        if operation == "average":

            if group_column:

                result = (
                    analysis_dataframe
                    .groupby(group_column)[target_column]
                    .mean()
                    .reset_index()
                )

                result = result.rename(
                    columns={
                        target_column:
                        f"average_{target_column}"
                    }
                )

                return format_result(
                    result,
                    dataset_name,
                    f"Average {target_column}"
                )

            result = analysis_dataframe[
                target_column
            ].mean()

            return (
                f"Datasets: {dataset_name}\n\n"
                f"Average {target_column}: "
                f"{result:,.2f}"
            )

    except Exception as error:

        return (
            "I could not analyze the "
            "selected dataset(s).\n\n"
            f"Error: {str(error)}"
        )

    return None


def format_result(
    dataframe,
    dataset_name,
    title
):

    return (
        f"Datasets: {dataset_name}\n\n"
        f"{title}:\n\n"
        f"{dataframe.to_string(index=False)}"
    )