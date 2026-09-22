from __future__ import annotations

import pandas as pd

from services.query_engine import QueryEngine


class ReportService:
    """
    Compatibility wrapper around QueryEngine.

    ReportService is kept so existing code can continue
    importing it without breaking.
    """

    @staticmethod
    def generate_report(
        dataset,
        dimensions=None,
        metrics=None,
        filters=None,
        sort_by=None,
        sort_direction="desc",
        limit=None,
    ) -> pd.DataFrame:

        dimensions = dimensions or []
        metrics = metrics or []
        filters = filters or {}

        query_filters = []

        for column, value in filters.items():

            if isinstance(value, list):

                query_filters.append(
                    {
                        "column": column,
                        "operator": "in",
                        "value": value,
                    }
                )

            else:

                query_filters.append(
                    {
                        "column": column,
                        "operator": "=",
                        "value": value,
                    }
                )

        query = {
            "columns": dimensions or ["*"],
            "filters": query_filters,
            "group_by": dimensions if metrics else [],
            "aggregations": metrics,
            "order_by": [],
            "limit": limit,
        }

        if sort_by:

            query["order_by"] = [
                {
                    "column": sort_by,
                    "direction": sort_direction,
                }
            ]

        return QueryEngine.execute(
            dataset,
            query
        )