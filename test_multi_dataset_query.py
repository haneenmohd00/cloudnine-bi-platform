from app import app
from models.dataset import Dataset

from services.query_engine import QueryEngine


with app.app_context():

    raffi = Dataset.query.filter(
        Dataset.original_filename.ilike(
            "%sample-raffi%"
        )
    ).first()

    centre = Dataset.query.filter(
        Dataset.original_filename.ilike(
            "%centre_data%"
        )
    ).first()

    if not raffi:
        raise RuntimeError(
            "sample-raffi.xlsx not found."
        )

    if not centre:
        raise RuntimeError(
            "centre_data.csv not found."
        )

    datasets = [
        raffi,
        centre
    ]

    # =====================================================
    # TEST 1
    # Join Raffi → Centre Data
    # =====================================================

    query = {

        "base_dataset_id": raffi.id,

        "joins": [
            {
                "dataset_id": centre.id,
                "left_column": "site",
                "right_column": "site",
                "how": "left"
            }
        ],

        "columns": [
            "site",
            "Total Net Revenue",
            "center",
            "regional_director"
        ],

        "filters": [],

        "group_by": [
            "site",
            "center",
            "regional_director"
        ],

        "aggregations": [
            {
                "column": "net",
                "aggregation": "sum",
                "alias": "Total Net Revenue"
            }
        ],

        "order_by": [
            {
                "column": "Total Net Revenue",
                "direction": "desc"
            }
        ],

        "limit": 10
    }

    result = QueryEngine.execute_multi_dataset(
        datasets,
        query
    )

    print("\n")
    print("=" * 80)
    print("MULTI-DATASET JOIN TEST")
    print("=" * 80)

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nRows returned: {len(result)}"
    )