from pathlib import Path
import uuid
import time
import pandas as pd

from markdown import markdown
import json
import os
import hashlib
import io

from services.llm_service import (
    ask_llm,
    generate_business_summary
)
from services.analytics_service import analyze_question
from services.overview_service import (
    generate_overview,
    build_ai_summary_context
)
from services.data_service import DataService
from services.dataset_context_service import (
    build_dataset_context
)

from services.dashboard_service import DashboardService
from services.relationship_service import (
    RelationshipService
)
from services.ai_orchestrator import AIOrchestrator
from services.report_ai_service import ReportAIService
from services.query_engine import QueryEngine

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    Response,
    send_file
)

from werkzeug.utils import secure_filename

from extensions import db
from models.dataset import Dataset


main_bp = Blueprint("main", __name__)



ai_orchestrator = AIOrchestrator()
report_ai_service = ReportAIService()

@main_bp.route("/")
def home():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    selected_datasets = []

    if selected_dataset_ids:

        selected_datasets = Dataset.query.filter(
            Dataset.id.in_(selected_dataset_ids)
        ).all()

    # ---------------------------------------------
    # NORMAL BUSINESS OVERVIEW
    # ---------------------------------------------

    overview = generate_overview(
        selected_datasets
    )

    # ---------------------------------------------
    # CHECK CACHED AI SUMMARY ONLY
    #
    # IMPORTANT:
    # Never call Ollama while loading Home.
    # ---------------------------------------------

    ai_summary = None

    if selected_datasets:

        try:

            cache_data = {
                "dataset_ids":
                    sorted(selected_dataset_ids),

                "overview":
                    overview
            }

            cache_string = json.dumps(
                cache_data,
                sort_keys=True,
                default=str
            )

            cache_key = hashlib.md5(
                cache_string.encode("utf-8")
            ).hexdigest()

            cache_dir = os.path.join(
                "instance",
                "cache"
            )

            os.makedirs(
                cache_dir,
                exist_ok=True
            )

            cache_file = os.path.join(
                cache_dir,
                "ai_summary.json"
            )

            if os.path.exists(cache_file):

                with open(
                    cache_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    cache = json.load(file)

                if cache.get(
                    "cache_key"
                ) == cache_key:

                    ai_summary = cache.get(
                        "summary"
                    )

        except Exception as error:

            print(
                "AI cache error:",
                error
            )

    # ---------------------------------------------
    # RENDER EXISTING CACHED SUMMARY
    # ---------------------------------------------

    ai_summary_html = None

    if ai_summary:

        ai_summary_html = markdown(
            ai_summary
        )

    return render_template(
        "home.html",
        overview=overview,
        selected_datasets=selected_datasets,
        ai_summary=ai_summary_html
    )

@main_bp.route("/home/ai-summary",methods=["POST"])
def home_ai_summary():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    if not selected_dataset_ids:

        return {
            "success": False,
            "message": "Please select a dataset first."
        }, 400

    selected_datasets = Dataset.query.filter(
        Dataset.id.in_(selected_dataset_ids)
    ).all()

    try:

        # -----------------------------------------
        # CALCULATE VALIDATED BUSINESS DATA
        # -----------------------------------------

        overview = generate_overview(
            selected_datasets
        )

        # -----------------------------------------
        # CACHE KEY
        # -----------------------------------------

        cache_data = {
            "dataset_ids":
                sorted(selected_dataset_ids),

            "overview":
                overview
        }

        cache_string = json.dumps(
            cache_data,
            sort_keys=True,
            default=str
        )

        cache_key = hashlib.md5(
            cache_string.encode("utf-8")
        ).hexdigest()

        cache_dir = os.path.join(
            "instance",
            "cache"
        )

        os.makedirs(
            cache_dir,
            exist_ok=True
        )

        cache_file = os.path.join(
            cache_dir,
            "ai_summary.json"
        )

        # -----------------------------------------
        # RETURN CACHE IF AVAILABLE
        # -----------------------------------------

        if os.path.exists(cache_file):

            with open(
                cache_file,
                "r",
                encoding="utf-8"
            ) as file:

                cache = json.load(file)

            if cache.get(
                "cache_key"
            ) == cache_key:

                cached_summary = cache.get(
                    "summary"
                )

                if cached_summary:

                    return {
                        "success": True,
                        "summary": markdown(
                            cached_summary
                        ),
                        "cached": True
                    }

        # -----------------------------------------
        # BUILD CONTROLLED AI CONTEXT
        # -----------------------------------------

        summary_context = (
            build_ai_summary_context(
                overview
            )
        )

        # -----------------------------------------
        # GENERATE AI SUMMARY
        # -----------------------------------------

        raw_summary = (
            generate_business_summary(
                summary_context
            )
        )

        # -----------------------------------------
        # SAVE TO CACHE
        # -----------------------------------------

        with open(
            cache_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "cache_key":
                        cache_key,

                    "summary":
                        raw_summary
                },
                file,
                ensure_ascii=False,
                indent=2
            )

        return {
            "success": True,
            "summary": markdown(
                raw_summary
            ),
            "cached": False
        }

    except Exception as error:

        print(
            "Home AI summary error:",
            error
        )

        return {
            "success": False,
            "message":
                "AI business analysis is currently unavailable."
        }, 500


@main_bp.route("/datasets")
def datasets():

    dataset_list = Dataset.query.order_by(
        Dataset.uploaded_at.desc()
    ).all()

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    return render_template(
        "datasets.html",
        datasets=dataset_list,
        selected_dataset_ids=selected_dataset_ids
    )

@main_bp.route("/datasets/select", methods=["POST"])
def select_datasets():

    selected_dataset_ids = request.form.getlist(
        "dataset_ids"
    )

    selected_dataset_ids = [
        int(dataset_id)
        for dataset_id in selected_dataset_ids
    ]

    session["selected_dataset_ids"] = selected_dataset_ids

    flash(
        f"{len(selected_dataset_ids)} dataset(s) selected for analysis."
    )

    return redirect(
        url_for("main.dataset_relationships")
    )

@main_bp.route("/datasets/relationships")
def dataset_relationships():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    if len(selected_dataset_ids) < 2:

        return render_template(
            "dataset_relationships.html",
            relationships=[],
            selected_datasets=[]
        )

    selected_datasets = Dataset.query.filter(
        Dataset.id.in_(selected_dataset_ids)
    ).all()

    relationships = (
        RelationshipService.detect_relationships(
            selected_datasets
        )
    )

    return render_template(
        "dataset_relationships.html",
        relationships=relationships,
        selected_datasets=selected_datasets
    )

@main_bp.route("/datasets/clear-selection", methods=["POST"])
def clear_dataset_selection():

    session.pop(
        "selected_dataset_ids",
        None
    )

    flash("Dataset selection cleared.")

    return redirect(
        url_for("main.datasets")
    )

@main_bp.route("/datasets/delete/<int:dataset_id>",methods=["POST"])
def delete_dataset(dataset_id):

    dataset = Dataset.query.get_or_404(
        dataset_id
    )

    # Remove dataset ID from current selection
    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    if dataset_id in selected_dataset_ids:

        selected_dataset_ids.remove(
            dataset_id
        )

        session[
            "selected_dataset_ids"
        ] = selected_dataset_ids

    # Remove physical uploaded file
    if dataset.file_path:

        file_path = Path(
            dataset.file_path
        )

        if file_path.exists():

            file_path.unlink()

    # Remove database record
    db.session.delete(
        dataset
    )

    db.session.commit()

    flash(
        f"Dataset '{dataset.source_name}' deleted successfully."
    )

    return redirect(
        url_for("main.datasets")
    )

@main_bp.route("/upload", methods=["GET", "POST"])
def upload_dataset():

    if request.method == "POST":

        if "dataset" not in request.files:
            flash("No file selected.")
            return redirect(request.url)

        file = request.files["dataset"]

        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not DataService.is_allowed_file(file.filename):
            flash("Only CSV and Excel files are supported.")
            return redirect(request.url)

        original_filename = secure_filename(file.filename)

        extension = Path(original_filename).suffix.lower()

        unique_filename = (
            f"{uuid.uuid4().hex}{extension}"
        )

        upload_folder = Path("uploads")
        upload_folder.mkdir(exist_ok=True)

        file_path = upload_folder / unique_filename

        file.save(file_path)

        try:
            dataframe = DataService.load_dataset(file_path)

            metadata = DataService.get_dataset_metadata(
                dataframe
            )

            dataset = Dataset(
            source_type="file",
            source_name=original_filename,

            original_filename=original_filename,
            stored_filename=unique_filename,
            file_path=str(file_path),
            file_type=extension,

            row_count=metadata["rows"],
            column_count=metadata["columns_count"]
            )

            db.session.add(dataset)
            db.session.commit()

        except Exception as error:

            if file_path.exists():
                file_path.unlink()

            flash(
                f"Unable to process dataset: {error}"
            )

            return redirect(
                url_for("main.upload_dataset")
            )

        return redirect(
            url_for(
                "main.dataset_preview",
                dataset_id=dataset.id
            )
        )

    return render_template("upload.html")


@main_bp.route("/dataset/<int:dataset_id>")
def dataset_preview(dataset_id):

    dataset = Dataset.query.get_or_404(dataset_id)

    dataframe = DataService.load_dataset(
        dataset.file_path
    )

    metadata = DataService.get_dataset_metadata(
        dataframe
    )

    return render_template(
        "dataset_preview.html",
        dataset=dataset,
        metadata=metadata
    )


@main_bp.route("/ai-assistant", methods=["GET", "POST"])
def ai_assistant():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    selected_datasets = (
        Dataset.query
        .filter(
            Dataset.id.in_(selected_dataset_ids)
        )
        .all()
        if selected_dataset_ids
        else []
    )

    question = ""
    answer = None

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if question:

            try:

                answer = ai_orchestrator.answer_question(
                    question=question,
                    selected_datasets=selected_datasets
                )

            except Exception as error:

                answer = (
                    "Sorry, I could not process your "
                    f"request. Error: {error}"
                )

    return render_template(
        "ai_assistant.html",
        question=question,
        answer=answer,
        selected_datasets=selected_datasets
    )


@main_bp.route("/dashboard")
def dashboard():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    selected_datasets = []

    if selected_dataset_ids:

        selected_datasets = Dataset.query.filter(
            Dataset.id.in_(selected_dataset_ids)
        ).all()

    dashboard_data = (
        DashboardService.analyze_selected_datasets(
            selected_datasets
        )
    )

    return render_template(
        "dashboard.html",
        selected_datasets=selected_datasets,
        dashboard_data=dashboard_data
    )

@main_bp.route("/dashboard/ai-analysis", methods=["POST"])
def dashboard_ai_analysis():

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    if not selected_dataset_ids:

        return {
            "success": False,
            "message": "Please select a dataset first."
        }, 400


    selected_datasets = Dataset.query.filter(
        Dataset.id.in_(selected_dataset_ids)
    ).all()


    try:

        # ==========================================
        # GET THE SAME DATA USED BY THE DASHBOARD
        # ==========================================

        dashboard_data = (
            DashboardService.analyze_selected_datasets(
                selected_datasets
            )
        )


        # ==========================================
        # BUILD CONTROLLED AI CONTEXT
        # ==========================================

        dashboard_context = []


        for data in dashboard_data:

            if data.get("error"):

                continue


            dataset_info = {

                "dataset_name":
                    data.get("dataset_name"),

                "total_revenue":
                    data.get("total_revenue"),

                "transactions":
                    data.get("rows"),

                "average_transaction":
                    data.get("average_transaction"),

                "total_quantity":
                    data.get("total_quantity"),

                "revenue_column":
                    data.get("revenue_column"),

                "dimension_column":
                    data.get("dimension_column"),

                "dimension_performance":
                    data.get("dimension_performance"),

                "time_performance":
                    data.get("time_performance")

            }


            dashboard_context.append(
                dataset_info
            )


        # ==========================================
        # SEND VALIDATED DATA TO AI
        # ==========================================

        prompt = f"""
You are a senior business intelligence analyst.

You are analysing a business dashboard.

The numbers below were calculated by the application's
analytics engine from the user's selected dataset.

DO NOT invent numbers.
DO NOT calculate unsupported metrics.
DO NOT assume missing business information.
Only use the supplied dashboard data.

Your job is NOT to simply repeat the KPI numbers.

Explain what the dashboard means from a business
perspective.

Identify:

1. Overall business performance
2. The strongest performing area
3. Important revenue concentration or distribution
4. Important trends, if multiple time periods exist
5. Areas that deserve management attention
6. Practical business actions or areas to investigate

The response should be concise but useful for a business
manager.

Use a professional executive-summary style.

Do not mention that you are an AI.

Do not use Markdown headings with ###.
Do not use excessive bullet points.

Write approximately 3-6 short paragraphs.

DASHBOARD DATA:

{json.dumps(
    dashboard_context,
    indent=2,
    default=str
)}
"""


        # ==========================================
        # CALL LOCAL LLM
        # ==========================================

        raw_summary = ask_llm(
            question=prompt,
            dataset_context=""
        )


        # ==========================================
        # RENDER MARKDOWN SAFELY
        # ==========================================

        summary_html = markdown(
            raw_summary
        )


        return {

            "success": True,

            "summary":
                summary_html

        }


    except Exception as error:

        print(
            "Dashboard AI analysis error:",
            error
        )


        return {

            "success": False,

            "message":
                "AI analysis is currently unavailable."

        }, 500

@main_bp.route("/insights")
def insights():
    return render_template(
        "placeholder.html",
        page_title="Insights",
        message="AI-generated and data-driven insights will appear here."
    )

# *********************************************************************************

def build_report_query_summary(query):
    """
    Convert the structured query specification into a
    human-readable description for the Report Generator UI.
    """

    if not query:
        return None

    parts = []

    # ---------------------------------------------------------
    # Selected columns
    # ---------------------------------------------------------

    columns = query.get(
        "columns",
        ["*"]
    )

    if columns == ["*"]:

        parts.append(
            "all available columns"
        )

    elif columns:

        formatted_columns = ", ".join(
            str(column)
            for column in columns
        )

        parts.append(
            f"columns: {formatted_columns}"
        )

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------

    filters = query.get(
        "filters",
        []
    )

    if filters:

        filter_text = []

        operator_labels = {
            "=": "equals",
            "==": "equals",
            "!=": "does not equal",
            ">": "is greater than",
            ">=": "is greater than or equal to",
            "<": "is less than",
            "<=": "is less than or equal to",
            "contains": "contains",
            "not_contains": "does not contain",
            "starts_with": "starts with",
            "ends_with": "ends with",
            "in": "is one of",
            "not_in": "is not one of",
            "is_null": "is empty",
            "is_not_null": "is not empty",
            "between": "is between",
        }

        for condition in filters:

            column = condition.get(
                "column"
            )

            operator = condition.get(
                "operator",
                "="
            )

            value = condition.get(
                "value"
            )

            label = operator_labels.get(
                operator,
                operator
            )

            if operator in {
                "is_null",
                "is_not_null"
            }:

                filter_text.append(
                    f"{column} {label}"
                )

            else:

                if isinstance(value, list):

                    formatted_value = ", ".join(
                        str(item)
                        for item in value
                    )

                else:

                    formatted_value = str(
                        value
                    )

                filter_text.append(
                    f"{column} {label} "
                    f"{formatted_value}"
                )

        parts.append(
            "where " +
            " and ".join(filter_text)
        )

    # ---------------------------------------------------------
    # Grouping
    # ---------------------------------------------------------

    group_by = query.get(
        "group_by",
        []
    )

    if group_by:

        parts.append(
            "grouped by " +
            ", ".join(group_by)
        )

    # ---------------------------------------------------------
    # Aggregations
    # ---------------------------------------------------------

    aggregations = query.get(
        "aggregations",
        []
    )

    if aggregations:

        aggregation_text = []

        for aggregation in aggregations:

            column = aggregation.get(
                "column"
            )

            function = aggregation.get(
                "aggregation",
                "sum"
            )

            alias = aggregation.get(
                "alias"
            )

            function_labels = {
                "sum": "total",
                "mean": "average",
                "average": "average",
                "min": "minimum",
                "max": "maximum",
                "count": "count",
                "nunique": "unique count",
            }

            function_label = (
                function_labels.get(
                    function,
                    function
                )
            )

            if alias:

                aggregation_text.append(
                    alias
                )

            else:

                aggregation_text.append(
                    f"{function_label} "
                    f"{column}"
                )

        parts.append(
            "calculated: " +
            ", ".join(
                aggregation_text
            )
        )

    # ---------------------------------------------------------
    # Sorting
    # ---------------------------------------------------------

    order_by = query.get(
        "order_by",
        []
    )

    if order_by:

        order_text = []

        for order in order_by:

            column = order.get(
                "column"
            )

            direction = order.get(
                "direction",
                "asc"
            )

            direction_label = (
                "highest to lowest"
                if direction == "desc"
                else "lowest to highest"
            )

            order_text.append(
                f"{column} "
                f"({direction_label})"
            )

        parts.append(
            "sorted by " +
            ", ".join(order_text)
        )

    # ---------------------------------------------------------
    # Limit
    # ---------------------------------------------------------

    limit = query.get(
        "limit"
    )

    if limit:

        parts.append(
            f"limited to {limit} records"
        )

    # ---------------------------------------------------------
    # Final sentence
    # ---------------------------------------------------------

    if not parts:
        return "Report generated from the selected dataset."

    return (
        "This report shows "
        + "; ".join(parts)
        + "."
    )


@main_bp.route("/report-generator", methods=["GET", "POST"])
def report_generator():

    PAGE_SIZE = 50
    DISPLAY_LIMIT = 500

    selected_dataset_ids = session.get(
        "selected_dataset_ids",
        []
    )

    selected_datasets = (
        Dataset.query
        .filter(
            Dataset.id.in_(selected_dataset_ids)
        )
        .all()
        if selected_dataset_ids
        else []
    )

    question = ""
    result_columns = []
    result_rows = []

    generated_query = None
    selected_dataset_id = None

    error = None

    total_rows = 0
    displayed_rows = 0
    total_pages = 0
    current_page = 1

    query_summary = None
    dataset_name = None

    # =========================================================
    # GET / POST
    # =========================================================

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        selected_dataset_id = request.form.get(
            "dataset_id",
            ""
        ).strip()

        if not question:

            error = (
                "Please describe the report "
                "you want to generate."
            )

        elif not selected_dataset_id:

            error = "Please select a dataset."

        else:

            try:

                selected_dataset_id = int(
                    selected_dataset_id
                )

                dataset = Dataset.query.get(
                    selected_dataset_id
                )

                if not dataset:

                    raise ValueError(
                        "Selected dataset was not found."
                    )

                dataset_name = (
                    dataset.original_filename
                    or dataset.source_name
                )

                # -------------------------------------------------
                # Natural language → Query specification
                # -------------------------------------------------

                generated_query = (
                    report_ai_service.generate_query(
                        question,
                        dataset
                    )
                )

                # -------------------------------------------------
                # Query specification → DataFrame
                # -------------------------------------------------

                result = QueryEngine.execute(
                    dataset,
                    generated_query
                )

                # -------------------------------------------------
                # Store report state
                # -------------------------------------------------

                session["report_dataset_id"] = (
                    dataset.id
                )

                session["report_query"] = (
                    generated_query
                )

                session["report_question"] = (
                    question
                )

                # Start from page 1
                current_page = 1

            except Exception as exc:

                error = str(exc)

    else:

        # =========================================================
        # Restore previous generated report on pagination
        # =========================================================

        selected_dataset_id = session.get(
            "report_dataset_id"
        )

        generated_query = session.get(
            "report_query"
        )

        question = session.get(
            "report_question",
            ""
        )

        if selected_dataset_id and generated_query:

            try:

                selected_dataset_id = int(
                    selected_dataset_id
                )

                dataset = Dataset.query.get(
                    selected_dataset_id
                )

                if dataset:

                    dataset_name = (
                        dataset.original_filename
                        or dataset.source_name
                    )

                    result = QueryEngine.execute(
                        dataset,
                        generated_query
                    )

                else:

                    result = None

            except Exception as exc:

                error = str(exc)
                result = None

        else:

            result = None

    # =========================================================
    # Process result
    # =========================================================

    if error is None and "result" in locals() and result is not None:

        # -----------------------------------------------------
        # Total result size
        # -----------------------------------------------------

        total_rows = len(result)

        # -----------------------------------------------------
        # Browser display limit
        # -----------------------------------------------------

        display_result = result.head(
            DISPLAY_LIMIT
        )

        displayed_rows = len(
            display_result
        )

        # -----------------------------------------------------
        # Pagination
        # -----------------------------------------------------

        try:

            requested_page = int(
                request.args.get(
                    "page",
                    1
                )
            )

        except (TypeError, ValueError):

            requested_page = 1

        max_display_page = max(
            1,
            (
                displayed_rows
                + PAGE_SIZE
                - 1
            ) // PAGE_SIZE
        )

        current_page = max(
            1,
            min(
                requested_page,
                max_display_page
            )
        )

        start_index = (
            current_page - 1
        ) * PAGE_SIZE

        end_index = (
            start_index + PAGE_SIZE
        )

        page_result = display_result.iloc[
            start_index:end_index
        ]

        result_columns = (
            page_result
            .columns
            .tolist()
        )

        result_rows = (
            page_result
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

        total_pages = max_display_page

        # -----------------------------------------------------
        # Human-readable query summary
        # -----------------------------------------------------

        query_summary = build_report_query_summary(
            generated_query
        )

    return render_template(
        "report_generator.html",

        selected_datasets=selected_datasets,

        selected_dataset_id=selected_dataset_id,

        question=question,

        result_columns=result_columns,

        result_rows=result_rows,

        generated_query=generated_query,

        row_count=total_rows,

        displayed_rows=displayed_rows,

        current_page=current_page,

        total_pages=total_pages,

        page_size=PAGE_SIZE,

        display_limit=DISPLAY_LIMIT,

        dataset_name=dataset_name,

        query_summary=query_summary,

        error=error,
    )

@main_bp.route("/report-generator/download/csv")
def download_report_csv():

    dataset_id = session.get(
        "report_dataset_id"
    )

    query = session.get(
        "report_query"
    )

    if not dataset_id or not query:

        return (
            "No generated report available.",
            400
        )

    dataset = Dataset.query.get(
        dataset_id
    )

    if not dataset:

        return (
            "Dataset not found.",
            404
        )

    try:

        result = QueryEngine.execute(
            dataset,
            query
        )

        csv_data = result.to_csv(
            index=False
        )

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename=cloudnine_report.csv"
            }
        )

    except Exception as error:

        return (
            f"Could not generate CSV: {error}",
            500
        )

@main_bp.route("/report-generator/download/excel")
def download_report_excel():

    dataset_id = session.get(
        "report_dataset_id"
    )

    query = session.get(
        "report_query"
    )

    if not dataset_id or not query:

        return (
            "No generated report available.",
            400
        )

    dataset = Dataset.query.get(
        dataset_id
    )

    if not dataset:

        return (
            "Dataset not found.",
            404
        )

    try:

        result = QueryEngine.execute(
            dataset,
            query
        )

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            result.to_excel(
                writer,
                index=False,
                sheet_name="Report"
            )

        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="cloudnine_report.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            )
        )

    except Exception as error:

        return (
            f"Could not generate Excel: {error}",
            500
        )