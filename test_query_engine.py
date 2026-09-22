from app import app
from models.dataset import Dataset

from services.query_engine import QueryEngine


with app.app_context():

    dataset = Dataset.query.filter(
        Dataset.original_filename.ilike(
            "%sample-raffi%"
        )
    ).first()

    if not dataset:

        raise RuntimeError(
            "sample-raffi.xlsx not found."
        )

    # ======================================================
    # TEST 1
    # All records from GZBINP
    # ======================================================

    query_1 = {

        "columns": ["*"],

        "filters": [
            {
                "column": "site",
                "operator": "=",
                "value": "GZBINP"
            }
        ],

        "limit": 20
    }

    result_1 = QueryEngine.execute(
        dataset,
        query_1
    )

    print("\n==============================")
    print("TEST 1: ALL GZBINP RECORDS")
    print("==============================")

    print(
        result_1.to_string(index=False)
    )

    print(
        f"\nRows returned: {len(result_1)}"
    )

    # ======================================================
    # TEST 2
    # Specific columns from GZBINP
    # ======================================================

    query_2 = {

        "columns": [
            "patient_name",
            "site",
            "item_name",
            "quantity",
            "net"
        ],

        "filters": [
            {
                "column": "site",
                "operator": "=",
                "value": "GZBINP"
            }
        ],

        "limit": 20
    }

    result_2 = QueryEngine.execute(
        dataset,
        query_2
    )

    print("\n==============================")
    print("TEST 2: SELECTED COLUMNS")
    print("==============================")

    print(
        result_2.to_string(index=False)
    )

    # ======================================================
    # TEST 3
    # Net greater than 1000
    # ======================================================

    query_3 = {

        "columns": [
            "site",
            "item_name",
            "net"
        ],

        "filters": [
            {
                "column": "net",
                "operator": ">",
                "value": 1000
            }
        ],

        "order_by": [
            {
                "column": "net",
                "direction": "desc"
            }
        ],

        "limit": 10
    }

    result_3 = QueryEngine.execute(
        dataset,
        query_3
    )

    print("\n==============================")
    print("TEST 3: NET > 1000")
    print("==============================")

    print(
        result_3.to_string(index=False)
    )

    # ======================================================
    # TEST 4
    # Revenue by site
    # ======================================================

    query_4 = {

        "columns": [
            "site",
            "Total Net Revenue"
        ],

        "group_by": [
            "site"
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

    result_4 = QueryEngine.execute(
        dataset,
        query_4
    )

    print("\n==============================")
    print("TEST 4: REVENUE BY SITE")
    print("==============================")

    print(
        result_4.to_string(index=False)
    )