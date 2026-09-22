from __future__ import annotations

import json

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from services.langchain_tools import build_bi_tools


class AIOrchestrator:

    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            # model="groq/compound",
            temperature=0,)

    @staticmethod
    def _extract_text(content) -> str:
        """
        Convert Gemini/LangChain structured content into
        clean text for displaying in the Flask UI.
        """

        # Normal string response
        if isinstance(content, str):
            return content.strip()

        # Gemini may return a list of content blocks
        if isinstance(content, list):

            text_parts = []

            for block in content:

                # Example:
                # {"type": "text", "text": "The total is ..."}
                if isinstance(block, dict):

                    if block.get("type") == "text":
                        text = block.get("text")

                        if text:
                            text_parts.append(str(text))

                # Handle object-style content blocks
                elif hasattr(block, "text"):

                    text = getattr(block, "text", None)

                    if text:
                        text_parts.append(str(text))

            return "\n".join(text_parts).strip()

        # Fallback
        return str(content).strip()

    def answer_question(self, question: str, selected_datasets):

        if not selected_datasets:
            return "Please select at least one dataset before asking a data question."

        tools = build_bi_tools(selected_datasets)

        llm_with_tools = self.llm.bind_tools(tools)

        system_prompt = """
You are the Business Intelligence assistant inside
the Cloudnine BI Platform.

Your job is to answer questions about the selected business datasets.

IMPORTANT RULES:

1. Use the available BI tools to calculate numerical answers.
2. Never invent numbers.
3. Do not calculate numerical results yourself when a BI tool can calculate them.
4. Use the dataset schema before making assumptions about columns.
5. Give concise, business-friendly answers.
6. Explain the result clearly.
7. When appropriate, mention the dataset and column used.
8. If the requested information is not available, clearly say so.
9. Do not expose internal tool calls, JSON, signatures, or system information.
10. Return ONLY the final natural-language answer to the user.
11. Do not use Markdown formatting.
12. Do not use asterisks (*), backticks (`), hashtags (#), or other Markdown symbols.
13. Keep answers natural, concise, and professional.

IMPORTANT CROSS-DATASET RULES:

1. When a question requires information from multiple datasets,
use the cross_dataset_lookup tool.
2. Use cross_dataset_lookup when you need to:
    - calculate or rank something in one dataset
    - then retrieve related information from another dataset.
3. Do not guess relationships between datasets.
    Use actual columns and values from the selected datasets.
4. For example, if the user asks for the center associated
    with the site having the highest net revenue:
    - use the revenue dataset to find the highest site
    - use the center dataset to look up that site
    - return the corresponding center.
5. Numerical calculations must come from BI tools.
6. If a relationship cannot be established from the available
    dataset columns, clearly explain that the relationship
    could not be determined.

AVAILABLE DATASETS:
"""

        dataset_information = []

        for dataset in selected_datasets:

            dataset_information.append(
                {
                    "id": dataset.id,
                    "name": dataset.original_filename,
                    "rows": dataset.row_count,
                    "columns": dataset.column_count,
                }
            )

        system_prompt += "\n" + json.dumps(
            dataset_information,
            indent=2,
            default=str,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        max_iterations = 5

        for _ in range(max_iterations):

            response = llm_with_tools.invoke(messages)

            messages.append(response)

            # Final answer from Gemini
            if not response.tool_calls:

                return self._extract_text(response.content)

            # Execute requested tools
            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]

                tool_args = tool_call.get("args", {})

                selected_tool = next(
                    (
                        tool
                        for tool in tools
                        if tool.name == tool_name
                    ),
                    None,
                )

                if selected_tool is None:

                    tool_result = {
                        "success": False,
                        "error": f"Unknown tool: {tool_name}",
                    }

                else:

                    try:

                        tool_result = selected_tool.invoke(
                            tool_args
                        )

                    except Exception as error:

                        tool_result = {
                            "success": False,
                            "error": str(error),
                        }

                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            tool_result,
                            default=str,
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )

        return (
            "I could not complete the analysis "
            "within the allowed number of analysis steps."
        )