from ollama import chat


def ask_llm(question, dataset_context=None):

    system_prompt = """
You are the Cloudnine Business Intelligence AI Assistant.

You help users analyze business data, understand KPIs,
generate insights, and answer business questions.

You may also answer general questions.

IMPORTANT RULES FOR DATASET QUESTIONS:

1. When dataset information is provided, use ONLY the actual
dataset names and column names provided in the dataset context.

2. Never invent, assume, or create hypothetical dataset columns.

3. If the user asks for available columns, list the actual
columns from the provided dataset context.

4. If a requested column does not exist, clearly tell the user
that the column is not available.

5. Do not claim that you calculated a value unless actual
calculation results were provided to you.

6. At this stage, dataset context contains schema information.
It does NOT mean you have access to all dataset values unless
those values are explicitly provided.
"""

    if dataset_context:
        system_prompt += f"""

Currently selected dataset information:

{dataset_context}
"""

    response = chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


def generate_business_summary(overview_context):

    system_prompt = """
You are the Cloudnine Business Intelligence AI Assistant.

Your task is to summarize calculated business performance
for a business user.

IMPORTANT RULES:

1. Use ONLY the numerical and business information provided
   in the context.

2. Do not invent numbers, datasets, trends, causes,
   anomalies, or recommendations that are not supported
   by the provided information.

3. Do not perform your own calculations.

4. Clearly distinguish facts from observations.

5. Keep the summary concise and business-focused.

6. Highlight the most important business observations.

7. If the available information is insufficient to determine
   something, say so instead of guessing.

Return a professional business summary suitable for the
Business Overview page.
"""

    response = chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": overview_context
            }
        ]
    )

    return response["message"]["content"]