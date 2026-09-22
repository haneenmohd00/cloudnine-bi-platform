from app import app
from models.dataset import Dataset

from services.report_ai_service import ReportAIService
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

    ai_service = ReportAIService()

    question = (
        "Give me patient name, item name, "
        "quantity and net for site GZBINP"
    )

    print("\nUSER QUESTION:")
    print(question)

    # ---------------------------------------------
    # Step 1: Natural language → Query
    # ---------------------------------------------

    query = ai_service.generate_query(
        question,
        dataset
    )

    print("\nGENERATED QUERY:")
    print(query)

    # ---------------------------------------------
    # Step 2: Query → DataFrame
    # ---------------------------------------------

    result = QueryEngine.execute(
        dataset,
        query
    )

    print("\nRESULT:")
    print(
        result.to_string(index=False)
    )

    print(
        f"\nRows returned: {len(result)}"
    )