import pandas as pd


def load_dataset(dataset):

    file_type = dataset.file_type.lower().lstrip(".")

    if file_type == "csv":
        return pd.read_csv(dataset.file_path)

    if file_type in ["xlsx", "xls"]:
        return pd.read_excel(dataset.file_path)

    raise ValueError(
        f"Unsupported file type: {dataset.file_type}"
    )


def generate_overview(datasets):

    overview = {
        "total_revenue": None,
        "total_transactions": None,
        "average_transaction": None,
        "total_quantity": None,
        "site_performance": None,
        "dataset_names": []
    }

    if not datasets:
        return overview

    frames = []

    for dataset in datasets:

        try:
            dataframe = load_dataset(dataset)

            overview["dataset_names"].append(
                dataset.source_name
            )

            dataframe["_source_dataset"] = (
                dataset.source_name
            )

            frames.append(dataframe)

        except Exception:
            continue

    if not frames:
        return overview

    # Combine compatible datasets
    dataframe = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    # -------------------------
    # Revenue
    # -------------------------

    if "net" in dataframe.columns:

        dataframe["net"] = pd.to_numeric(
            dataframe["net"],
            errors="coerce"
        )

        overview["total_revenue"] = (
            dataframe["net"].sum()
        )

        overview["average_transaction"] = (
            dataframe["net"].mean()
        )

    # -------------------------
    # Transactions
    # -------------------------

    if "trans_id" in dataframe.columns:

        overview["total_transactions"] = (
            dataframe["trans_id"].nunique()
        )

    else:

        overview["total_transactions"] = len(
            dataframe
        )

    # -------------------------
    # Quantity
    # -------------------------

    if "quantity" in dataframe.columns:

        dataframe["quantity"] = pd.to_numeric(
            dataframe["quantity"],
            errors="coerce"
        )

        overview["total_quantity"] = (
            dataframe["quantity"].sum()
        )

    # -------------------------
    # Site performance
    # -------------------------

    if (
        "site" in dataframe.columns
        and "net" in dataframe.columns
    ):

        site_data = (
            dataframe
            .dropna(subset=["site", "net"])
            .groupby("site")["net"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        overview["site_performance"] = (
            site_data.to_dict()
        )

    return overview


def build_ai_summary_context(overview):

    context = f"""
BUSINESS OVERVIEW DATA

Total Revenue:
{overview.get("total_revenue")}

Total Transactions:
{overview.get("total_transactions")}

Average Transaction:
{overview.get("average_transaction")}

Total Quantity:
{overview.get("total_quantity")}

Top Sites by Revenue:
"""

    site_performance = overview.get(
        "site_performance"
    )

    if site_performance:

        for site, revenue in site_performance.items():

            context += (
                f"- {site}: {revenue:,.2f}\n"
            )

    return context