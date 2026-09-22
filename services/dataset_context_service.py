import pandas as pd


def get_dataset_schema(dataset):
    """
    Reads a file-based dataset and returns its actual schema.
    """

    if dataset.source_type != "file":
        return (
            f"Dataset: {dataset.source_name}\n"
            f"Source Type: {dataset.source_type}\n"
            "Schema reading is not implemented yet for this source type."
        )

    try:
        file_path = dataset.file_path
        file_type = dataset.file_type.lower().lstrip(".")

        if file_type == "csv":
            dataframe = pd.read_csv(
                file_path,
                nrows=5
            )

        elif file_type in ["xlsx", "xls"]:
            dataframe = pd.read_excel(
                file_path,
                nrows=5
            )

        else:
            return (
                f"Dataset: {dataset.source_name}\n"
                f"Unsupported file type: {file_type}"
            )

        schema_text = (
            f"\nDataset Name: {dataset.source_name}\n"
            f"Source Type: {dataset.source_type}\n"
            f"Total Rows: {dataset.row_count}\n"
            f"Total Columns: {dataset.column_count}\n\n"
            f"Actual Columns and Data Types:\n"
        )

        for column in dataframe.columns:

            schema_text += (
                f"- {column} "
                f"(dtype: {dataframe[column].dtype})\n"
            )

        return schema_text

    except Exception as error:

        return (
            f"Dataset: {dataset.source_name}\n"
            f"Schema could not be read.\n"
            f"Error: {str(error)}"
        )


def build_dataset_context(datasets):
    """
    Builds schema context for one or multiple selected datasets.
    """

    if not datasets:
        return ""

    context = (
        "The user has selected the following datasets.\n"
        "Use ONLY the actual columns listed below when answering "
        "questions about the datasets.\n\n"
    )

    for dataset in datasets:

        context += get_dataset_schema(dataset)
        context += "\n"
        context += "=" * 50
        context += "\n"

    return context