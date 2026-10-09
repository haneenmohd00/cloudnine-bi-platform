# Cloudnine Hospitals — AI-Powered Business Intelligence Platform

**Turning business data into insights that are easier to explore, understand, and act on.**

A Flask-based Business Intelligence platform developed during my Data Engineering internship at Cloudnine Hospitals. The goal is to bring data exploration, interactive dashboards, and AI-assisted analysis into one place, so users can spend less time working through raw data and more time understanding what it means.

Rather than treating AI as a standalone chatbot, the platform connects it to actual dataset information and calculated results to help answer business questions more meaningfully.

<img width="1817" height="867" alt="ss5" src="https://github.com/user-attachments/assets/d9aba0e5-4d1b-4cce-a212-45a0c28041d0" />
<img width="1772" height="873" alt="ss2" src="https://github.com/user-attachments/assets/62c5ab50-665a-4b46-8dec-03c07c14cc77" />
<img width="1635" height="843" alt="ss1" src="https://github.com/user-attachments/assets/a8df95d2-e5e2-4cb1-b5db-ca9dd0711f31" />
<img width="1561" height="1078" alt="ss4" src="https://github.com/user-attachments/assets/915520a0-28b0-4d12-96ec-1f531f9446ef" />

## What I'm Building

The idea behind this project is simple: business users shouldn't always need to write SQL queries or manually analyse spreadsheets to find answers in their data.

The platform provides a starting point for exploring datasets, viewing key metrics, identifying trends, and asking questions in natural language. I'm building it as a modular application so its data-handling and AI components can evolve as the project grows.

## Features

- **Dataset Management** — Upload and work with CSV and Excel files, inspect their structure, preview records, and select datasets for analysis.
- **Dynamic Dashboards** — Generate KPI cards and visual summaries based on the available data instead of relying entirely on fixed column names.
- **Business Analytics** — Calculate totals, averages, counts, grouped metrics, Top-N results, and time-based trends.
- **Cross-Dataset Analysis** — Detect potential relationships between datasets and support safe joins where the available columns allow it.
- **AI Assistant** — Ask business questions in natural language and get assistance exploring the data.
- **AI-Generated Summaries** — Use calculated business metrics as context for AI-generated explanations and executive-style summaries.
- **Local LLM Support** — Use Qwen 2.5 through Ollama for selected AI features, allowing local model execution during development.
- **Modular Architecture** — Keep the Flask application, data processing, analytics, and AI logic organised into separate components.

## What Makes It Different?

Traditional dashboards are useful when you already know what you want to look at. This project explores what happens when you combine dashboards with a conversational way of investigating the underlying data.

The focus is on connecting AI responses to actual calculations rather than asking a language model to invent numbers. Python handles the analytical work, while the AI layer helps explain results in a more accessible way.

The platform is also being developed with flexibility in mind: datasets can have different schemas, and the application should be able to adapt to the data it receives.

## Tech Stack

| Area | Technologies |
|---|---|
| Backend | Python, Flask |
| Data Processing | Pandas, NumPy |
| Database | SQLite, SQLAlchemy |
| Visualisation | Plotly, HTML, CSS, JavaScript |
| AI / LLM | LangChain, Groq API, Ollama, Qwen 2.5 |
| Data Sources | CSV, Excel |
| Development | Git, GitHub, VS Code |

## How It Works

At a high level, the application follows this flow:

1. **Load data** from supported files.
2. **Inspect the schema** to identify useful columns, dimensions, dates, and metrics.
3. **Process the data** using Python to calculate relevant results.
4. **Present the results** through dashboards and analytical summaries.
5. **Use AI to explain** the available data and calculated results in natural language.

This separation between data processing and AI-generated explanations is an important part of the design.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies
- Git
- Access to the configured AI provider for features that require it
- Ollama and the configured Qwen model for locally powered AI features

### Installation

Clone the repository:

```bash
git clone <your-private-repository-url>
cd <your-repository-folder>
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the required environment variables using the project's configuration and `.env.example`, if provided. Never commit API keys or other secrets.

For local Qwen support, install Ollama separately, then pull the model configured by the project. For example:

```bash
ollama pull qwen2.5:7b
```

Start the Flask application using the entry-point file configured in the repository. For example, if the entry point is `app.py`:

```bash
python app.py
```

Open the local URL displayed in the terminal.

*Note: The exact Python version, environment variables, model name, and startup command should match the current repository configuration.*

## Project Status

This project is an actively developed prototype. Core work has focused on dataset handling, data-driven analytics, dynamic dashboards, and AI-assisted business analysis.

Future development may include additional database connectors, more advanced reporting, retrieval-augmented generation (RAG), and broader integrations. These are planned extensions, not features to assume are already available.

## A Note on Data

The application has been developed using sample datasets. Any testing involving healthcare or business information should use approved, appropriately protected data. Real patient information, credentials, API keys, and confidential company files should never be committed to the repository.

## Why This Project Matters to Me

Working on this project has given me the opportunity to explore the intersection of data engineering, business intelligence, and generative AI. It has also helped me think beyond building individual dashboards and focus on how data can be processed, validated, presented, and explained within one application.

The project is still evolving, and each iteration is an opportunity to make the platform more useful, reliable, and adaptable.
