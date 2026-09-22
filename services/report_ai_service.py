from __future__ import annotations

import json

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


class ReportAIService:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
        )

    @staticmethod
    def _extract_text(content):

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            parts = []

            for block in content:

                if isinstance(block, dict):

                    text = block.get("text")

                    if text:
                        parts.append(
                            str(text)
                        )

                elif hasattr(block, "text"):

                    text = getattr(
                        block,
                        "text",
                        None
                    )

                    if text:
                        parts.append(
                            str(text)
                        )

            return "\n".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def build_schema_context(datasets):

        from services.bi_tools import _load_dataset

        if not isinstance(
            datasets,
            (list, tuple)
        ):
            datasets = [datasets]

        schemas = []

        for dataset in datasets:

            dataframe = _load_dataset(
                dataset
            )

            schemas.append(
                {
                    "dataset_id": dataset.id,
                    "dataset_name": (
                        dataset.original_filename
                        or dataset.source_name
                    ),
                    "rows": len(dataframe),
                    "columns": (
                        dataframe
                        .columns
                        .tolist()
                    )
                }
            )

        return schemas

    def generate_query(
        self,
        question,
        datasets):

        schema = self.build_schema_context(
            datasets
        )

        system_prompt = f"""
You are the query planner for the Cloudnine BI Platform.

Your job is to convert a user's natural-language
data request into a structured QueryEngine specification.

You are NOT responsible for calculating results.

The Python QueryEngine will execute the specification.

DATASET:

{json.dumps(schema, indent=2)}

AVAILABLE QUERY OPERATIONS:

1. columns

Use ["*"] when the user requests all records
or all columns.

Use specific column names when the user explicitly
requests specific fields.

2. filters

Each filter must contain:

column
operator
value

Supported operators:

=
==
!=
>
>=
<
<=
contains
not_contains
starts_with
ends_with
in
not_in
is_null
is_not_null
between

3. group_by

Use this when the user requests grouped or
aggregated results.

4. aggregations

Each aggregation MUST contain:

{{
    "column": "actual_dataset_column",
    "aggregation": "sum|mean|average|min|max|count|nunique",
    "alias": "human readable result name"
}}

IMPORTANT:
Always use the key "aggregation".
Never use "function".

Example:

{{
    "column": "net",
    "aggregation": "sum",
    "alias": "Total Net Revenue"
}}

5. joins

Use joins when the requested columns are spread across
multiple datasets.

Each join MUST contain:

{{
    "dataset_id": 2,
    "left_column": "site",
    "right_column": "site",
    "how": "left"
}}

Supported join types:

left
inner
right
outer

IMPORTANT:

- Only join datasets when necessary.
- Only use columns that actually exist.
- Prefer a LEFT join when the user asks for records
  from the primary/base dataset and additional information
  from another dataset.
- Use a common column such as "site" when it exists
  in both datasets.
- Do not invent join keys.

6. order_by

Each item contains:

column
direction

direction must be:

asc
desc

7. limit

Use null when the user did not request a limit.

IMPORTANT RULES:

- Only use columns that exist in the dataset.
- Never invent column names.
- Never invent values.
- Do not calculate numerical results.
- Do not write SQL.
- Do not return explanations.
- Return ONLY valid JSON.
- Use ["*"] when the user asks for all records.
- If the user asks for specific columns, return only those columns.
- Use filters when the user specifies conditions.
- Use group_by and aggregations only when aggregation is requested.
- base_dataset_id must be the dataset containing the primary
  records or primary metric requested by the user.
- Every dataset_id in joins must correspond to one of the
  available datasets.
- If the requested information exists in only one dataset,
  do not create a join.

OUTPUT FORMAT:

{{
    "base_dataset_id": 3,
    "joins": [],
    "columns": ["*"],
    "filters": [],
    "group_by": [],
    "aggregations": [],
    "order_by": [],
    "limit": null
}}

OUTPUT RULES:

- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use code fences.
- Do not explain the query.
- Every aggregation MUST use the key "aggregation".
- Every aggregation MUST contain an alias.
- order_by must reference an actual output column.
- If an aggregation has an alias, order_by should use that alias.

Example 1:

User:
Show the top 10 sites by total net revenue.

Output:

{{
    "columns": ["site", "Total Net Revenue"],
    "filters": [],
    "group_by": ["site"],
    "aggregations": [
        {{
            "column": "net",
            "aggregation": "sum",
            "alias": "Total Net Revenue"
        }}
    ],
    "order_by": [
        {{
            "column": "Total Net Revenue",
            "direction": "desc"
        }}
    ],
    "limit": 10
}}

Example 2:

User:
Find the top 10 sites by total net revenue and show
their center and regional director.

Output:

{{
    "base_dataset_id": 3,
    "joins": [
        {{
            "dataset_id": 2,
            "left_column": "site",
            "right_column": "site",
            "how": "left"
        }}
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
        {{
            "column": "net",
            "aggregation": "sum",
            "alias": "Total Net Revenue"
        }}
    ],
    "order_by": [
        {{
            "column": "Total Net Revenue",
            "direction": "desc"
        }}
    ],
    "limit": 10
}}

"""

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=question
            ),
        ]

        response = self.llm.invoke(
            messages
        )

        content = self._extract_text(
            response.content
        )

        # Remove accidental Markdown fences
        content = content.strip()

        if content.startswith("```"):

            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:

            query = json.loads(
                content
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "The AI returned an invalid "
                f"query specification: {error}"
            )

        if not isinstance(query, dict):

            raise ValueError(
                "AI query specification must "
                "be a JSON object."
            )

        return query