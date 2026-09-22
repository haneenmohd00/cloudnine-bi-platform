from app import app
from models.dataset import Dataset

from services.bi_tools import cross_dataset_analysis


with app.app_context():

    raffi = Dataset.query.filter(
        Dataset.original_filename.ilike("%sample-raffi%")
    ).first()

    centre = Dataset.query.filter(
        Dataset.original_filename.ilike("%centre_data%")
    ).first()

    if not raffi:
        raise RuntimeError("sample-raffi.xlsx not found")

    if not centre:
        raise RuntimeError("centre_data.csv not found")

    result = cross_dataset_analysis(
        primary_dataset=raffi,
        primary_dimension="site",
        primary_metric="net",
        lookup_dataset=centre,
        lookup_key="site",
        return_column="center",
        aggregation="sum",
        ranking="highest",
        n=1,
    )

    print("\nCross-dataset result:")
    print(result)