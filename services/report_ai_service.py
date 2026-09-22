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
    def build_schema_context(dataset):

        from services.bi_tools import _load_dataset

        dataframe = _load_dataset(
            dataset
        )

        columns = dataframe.columns.tolist()

        return {
            "dataset_id": dataset.id,
            "dataset_name": dataset.original_filename,
            "columns": columns,
        }

    def generate_query(
        self,
        question,
        dataset):

        schema = self.build_schema_context(
            dataset
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

5. order_by

Each item contains:

column
direction

direction must be:

asc
desc

6. limit

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

OUTPUT FORMAT:

{{
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

Example:

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