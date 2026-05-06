from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PREFIX = OUTPUT_DIR / "page73_76"
RUN_LOG = OUTPUT_DIR / "page73_76_run_output.txt"

# Keep matplotlib cache in a writable location so image generation works in headless environments.
MPL_CACHE_DIR = OUTPUT_DIR / "page73_76_mplconfig"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RACE_ROWS = [
    (1, "Thunder Bolt", "Autumn Cup", 1, 92, "Turf"),
    (2, "Silver Wind", "Autumn Cup", 2, 88, "Turf"),
    (3, "Ocean Star", "Autumn Cup", 4, 82, "Turf"),
    (4, "Thunder Bolt", "Spring Mile", 2, 90, "Dirt"),
    (5, "Crimson Flash", "Spring Mile", 1, 94, "Dirt"),
    (6, "Silver Wind", "Spring Mile", 3, 85, "Dirt"),
    (7, "Ocean Star", "River Stakes", 2, 87, "Turf"),
    (8, "Crimson Flash", "River Stakes", 5, 79, "Turf"),
    (9, "Thunder Bolt", "River Stakes", 1, 95, "Turf"),
]

SCHEMA_TEXT = """
Table: race_records
Columns:
- id (INTEGER): unique row id
- uma_name (TEXT): horse name
- race_name (TEXT): race title
- rank (INTEGER): finishing rank, smaller is better
- speed_rating (INTEGER): internal performance score
- track (TEXT): Turf or Dirt
""".strip()


@dataclass
class QueryPlan:
    """A compact plan that mimics what an LLM-powered query engine would produce."""

    mode: str
    title: str
    sql: str | None
    pandas_steps: str | None
    chart_path: Path | None


def log(message: str, lines: List[str]) -> None:
    """Print and store a transcript so learners can see the runtime flow later."""
    print(message)
    lines.append(message)


def build_database() -> sqlite3.Connection:
    """Create a tiny in-memory database that stands in for a production SQL system."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE race_records (
            id INTEGER PRIMARY KEY,
            uma_name TEXT NOT NULL,
            race_name TEXT NOT NULL,
            rank INTEGER NOT NULL,
            speed_rating INTEGER NOT NULL,
            track TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO race_records (id, uma_name, race_name, rank, speed_rating, track) VALUES (?, ?, ?, ?, ?, ?)",
        RACE_ROWS,
    )
    conn.commit()
    return conn


def schema_linking(user_query: str, lines: List[str]) -> str:
    """
    In a real Text2SQL system an LLM would receive the relevant schema.
    Here we simply show the exact schema context being passed into the pipeline.
    """
    lowered = user_query.lower()
    hints = []
    if "race" in lowered or "cup" in lowered or "mile" in lowered or "stakes" in lowered:
        hints.append("race_name")
    if "best" in lowered or "average" in lowered or "rank" in lowered or "top" in lowered:
        hints.append("rank")
    if "horse" in lowered or "who" in lowered:
        hints.append("uma_name")
    if "speed" in lowered:
        hints.append("speed_rating")

    log("1. Schema linking", lines)
    log(f"   Query: {user_query}", lines)
    log(f"   Relevant columns: {', '.join(hints) if hints else 'general table context'}", lines)
    return SCHEMA_TEXT


def generate_sql(user_query: str, lines: List[str]) -> str:
    """Translate a limited set of natural-language requests into safe SELECT-only SQL."""
    lowered = user_query.lower()
    if "autumn cup" in lowered and ("top" in lowered or "finish" in lowered):
        sql = (
            "SELECT uma_name, rank, speed_rating "
            "FROM race_records "
            "WHERE race_name = 'Autumn Cup' "
            "ORDER BY rank ASC"
        )
    elif "best average rank" in lowered or ("average rank" in lowered and "horse" in lowered):
        sql = (
            "SELECT uma_name, ROUND(AVG(rank), 2) AS avg_rank, COUNT(*) AS races "
            "FROM race_records "
            "GROUP BY uma_name "
            "ORDER BY avg_rank ASC, races DESC"
        )
    else:
        sql = (
            "SELECT race_name, uma_name, rank, speed_rating "
            "FROM race_records "
            "ORDER BY race_name ASC, rank ASC"
        )

    log("2. SQL generation", lines)
    log(f"   Generated SQL: {sql}", lines)
    return sql


def validate_sql(sql: str) -> None:
    """Reject unsafe statements so the demo matches slide 74's read-only guidance."""
    blocked_keywords = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER")
    normalized = sql.upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed in this demo.")
    if any(keyword in normalized for keyword in blocked_keywords):
        raise ValueError("Unsafe SQL detected.")


def execute_sql(conn: sqlite3.Connection, sql: str, lines: List[str], limit: int = 10) -> pd.DataFrame:
    """Run the validated SQL and cap the result size to avoid overwhelming the user."""
    validate_sql(sql)
    limited_sql = f"{sql} LIMIT {limit}"
    log("3. Execute + validate", lines)
    log(f"   Running: {limited_sql}", lines)
    result = pd.read_sql_query(limited_sql, conn)
    if result.empty:
        log("   Result: no rows returned", lines)
    else:
        log(f"   Result rows: {len(result)}", lines)
    return result


def explain_results(result: pd.DataFrame, title: str, lines: List[str]) -> str:
    """Turn a result table into a conversational explanation, like the final LLM stage."""
    log("4. Explain results", lines)
    if result.empty:
        explanation = f"{title}: the query ran correctly, but no matching rows were found."
    elif "avg_rank" in result.columns:
        best = result.iloc[0]
        explanation = (
            f"{title}: {best['uma_name']} has the best average rank at {best['avg_rank']}, "
            f"based on {int(best['races'])} races."
        )
    else:
        winner = result.iloc[0]
        explanation = (
            f"{title}: {winner['uma_name']} leads the result set with rank {winner['rank']} "
            f"and speed rating {winner['speed_rating']}."
        )
    log(f"   {explanation}", lines)
    return explanation


def plan_pandas_analysis(title: str, chart_path: Path) -> QueryPlan:
    """Create a Text2Pandas plan for an in-memory analytics task."""
    return QueryPlan(
        mode="text2pandas",
        title=title,
        sql=None,
        pandas_steps=(
            "Load DataFrame -> group by track -> calculate average speed_rating -> "
            "sort descending -> plot bar chart"
        ),
        chart_path=chart_path,
    )


def run_pandas_analysis(dataframe: pd.DataFrame, plan: QueryPlan, lines: List[str]) -> pd.DataFrame:
    """Simulate Text2Pandas by operating directly on an in-memory DataFrame."""
    log("Text2Pandas path", lines)
    log(f"   Steps: {plan.pandas_steps}", lines)
    result = (
        dataframe.groupby("track", as_index=False)["speed_rating"]
        .mean()
        .rename(columns={"speed_rating": "avg_speed_rating"})
        .sort_values("avg_speed_rating", ascending=False)
    )
    log("   Aggregated in-memory DataFrame:", lines)
    log(result.to_string(index=False), lines)
    return result


def save_pipeline_diagram(path: Path) -> None:
    """Draw the four-step Query Engine pipeline from slide 74."""
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.axis("off")

    boxes = [
        ("Schema\nLinking", "#2dd4bf"),
        ("SQL\nGeneration", "#60a5fa"),
        ("Execute\n+ Validate", "#f59e0b"),
        ("Explain\nResults", "#f43f5e"),
    ]
    positions = [0.04, 0.29, 0.54, 0.79]

    for x, (label, color) in zip(positions, boxes):
        rect = plt.Rectangle((x, 0.18), 0.17, 0.58, facecolor="#0f172a", edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x + 0.085, 0.47, label, ha="center", va="center", color="white", fontsize=13, weight="bold")

    for start in [0.215, 0.465, 0.715]:
        ax.annotate(
            "",
            xy=(start + 0.06, 0.47),
            xytext=(start, 0.47),
            arrowprops={"arrowstyle": "->", "lw": 2.3, "color": "#94a3b8"},
        )

    ax.text(0.5, 0.9, "Query Engine Flow", ha="center", va="center", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sql_chart(result: pd.DataFrame, path: Path) -> None:
    """Visualize SQL output so the lesson is not just text and tables."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(result["uma_name"], result["speed_rating"], color=["#0ea5e9", "#38bdf8", "#7dd3fc"])
    ax.set_title("Autumn Cup Speed Ratings", fontsize=14, weight="bold")
    ax.set_ylabel("Speed rating")
    ax.set_xlabel("Horse")
    ax.grid(axis="y", alpha=0.25)

    for bar, rank in zip(bars, result["rank"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"rank {rank}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pandas_chart(result: pd.DataFrame, path: Path) -> None:
    """Show where Text2Pandas shines: quick in-memory analytics with a chart."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(result["track"], result["avg_speed_rating"], color=["#22c55e", "#86efac"])
    ax.set_title("Average Speed Rating by Track", fontsize=14, weight="bold")
    ax.set_ylabel("Average speed rating")
    ax.set_xlabel("Track")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, result["avg_speed_rating"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.1f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_mode_comparison_chart(path: Path) -> None:
    """Summarize slide 76 by comparing when Text2SQL and Text2Pandas fit best."""
    categories = ["Production DB", "Joins", "Audit trail", "Quick charts", "Custom transforms", "Sandboxed logic"]
    text2sql_scores = [5, 5, 5, 2, 2, 3]
    text2pandas_scores = [2, 2, 2, 5, 5, 5]

    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(x, text2sql_scores, marker="o", linewidth=2.5, color="#2563eb", label="Text2SQL")
    ax.plot(x, text2pandas_scores, marker="o", linewidth=2.5, color="#16a34a", label="Text2Pandas")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=18, ha="right")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Better fit score")
    ax.set_title("When to Use Which", fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def log_table(title: str, dataframe: pd.DataFrame, lines: List[str]) -> None:
    """Write DataFrame contents into the transcript in a readable way."""
    log(title, lines)
    if dataframe.empty:
        log("   <empty>", lines)
    else:
        for row in dataframe.to_string(index=False).splitlines():
            log(f"   {row}", lines)


def ensure_parent(paths: Iterable[Path]) -> None:
    """Create directories for any output paths before saving files."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    lines: List[str] = []
    conn = build_database()
    dataframe = pd.DataFrame(
        RACE_ROWS,
        columns=["id", "uma_name", "race_name", "rank", "speed_rating", "track"],
    )

    pipeline_path = OUTPUT_DIR / "page73_76_query_engine_pipeline.png"
    sql_chart_path = OUTPUT_DIR / "page73_76_text2sql_result.png"
    pandas_chart_path = OUTPUT_DIR / "page73_76_text2pandas_result.png"
    comparison_chart_path = OUTPUT_DIR / "page73_76_mode_comparison.png"
    ensure_parent([pipeline_path, sql_chart_path, pandas_chart_path, comparison_chart_path, RUN_LOG])

    log("Query Engine demo for slides 73-76", lines)
    log("", lines)

    user_query = "Show the top finishers in Autumn Cup"
    schema = schema_linking(user_query, lines)
    log("   Schema passed to the generator:", lines)
    for row in schema.splitlines():
        log(f"   {row}", lines)
    sql = generate_sql(user_query, lines)
    sql_result = execute_sql(conn, sql, lines)
    log_table("   SQL result table:", sql_result, lines)
    explain_results(sql_result, "Autumn Cup summary", lines)
    log("", lines)

    second_query = "Which horse has the best average rank?"
    schema_linking(second_query, lines)
    avg_sql = generate_sql(second_query, lines)
    avg_result = execute_sql(conn, avg_sql, lines)
    log_table("   Average-rank result table:", avg_result, lines)
    explain_results(avg_result, "Average rank summary", lines)
    log("", lines)

    pandas_plan = plan_pandas_analysis(
        title="Average speed rating by track",
        chart_path=pandas_chart_path,
    )
    pandas_result = run_pandas_analysis(dataframe, pandas_plan, lines)
    log("", lines)

    save_pipeline_diagram(pipeline_path)
    save_sql_chart(sql_result, sql_chart_path)
    save_pandas_chart(pandas_result, pandas_chart_path)
    save_mode_comparison_chart(comparison_chart_path)

    log("Saved visuals:", lines)
    log(f"   - {pipeline_path.name}", lines)
    log(f"   - {sql_chart_path.name}", lines)
    log(f"   - {pandas_chart_path.name}", lines)
    log(f"   - {comparison_chart_path.name}", lines)

    RUN_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    conn.close()


if __name__ == "__main__":
    main()
