"""
page135_140_mlops_agent_demo.py

Educational simulation for slides 135-140:
- MLOps Agent pipeline: Planning -> Tool Execution -> Reflection -> Finalizing
- Agent framework comparison: LangGraph, CrewAI, AutoGen

Run:
    python3 page135_140/page135_140_mlops_agent_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("page135_140")
RANDOM_SEED = 7


@dataclass
class Task:
    """Represents one unit of work in the MLOps plan graph."""

    name: str
    depends_on: List[str]
    estimated_minutes: int
    tool: str


@dataclass
class StageLog:
    """Stores details collected during each pipeline stage."""

    planning: Dict = field(default_factory=dict)
    execution: Dict = field(default_factory=dict)
    reflection: Dict = field(default_factory=dict)
    finalizing: Dict = field(default_factory=dict)


# ------------------------------
# 1) Planning stage
# ------------------------------

def build_task_graph() -> List[Task]:
    """Create a small dependency graph like the slide's planning stage."""
    return [
        Task("collect_data", [], 20, "data_loader"),
        Task("clean_data", ["collect_data"], 25, "data_cleaner"),
        Task("feature_engineering", ["clean_data"], 25, "feature_builder"),
        Task("train_model", ["feature_engineering"], 35, "trainer"),
        Task("evaluate_model", ["train_model"], 20, "evaluator"),
        Task("package_artifacts", ["evaluate_model"], 10, "packager"),
    ]


def summarize_plan(tasks: List[Task]) -> Dict:
    total_minutes = sum(t.estimated_minutes for t in tasks)
    parallel_groups = [["collect_data"], ["clean_data"], ["feature_engineering"], ["train_model"], ["evaluate_model"], ["package_artifacts"]]
    return {
        "task_count": len(tasks),
        "total_estimated_minutes": total_minutes,
        "parallel_groups": parallel_groups,
        "priority_order": [t.name for t in tasks],
    }


# ------------------------------
# 2) Tool execution stage
# ------------------------------

def simulate_tool_execution(tasks: List[Task]) -> Dict:
    """Simulate runtime, success, and quality score for each task."""
    rng = random.Random(RANDOM_SEED)
    results = []

    for task in tasks:
        runtime = max(2, int(task.estimated_minutes * rng.uniform(0.6, 1.4)))
        quality = round(rng.uniform(0.55, 0.95), 3)
        success = quality >= 0.65

        results.append(
            {
                "task": task.name,
                "tool": task.tool,
                "runtime_minutes": runtime,
                "quality": quality,
                "success": success,
            }
        )

    return {"runs": results}


# ------------------------------
# 3) Reflection / self-correct stage
# ------------------------------

def reflect_and_retry(execution: Dict) -> Dict:
    """Detect weak outputs and retry with adjusted strategy."""
    retries = []
    fixed_runs = []

    for run in execution["runs"]:
        current = dict(run)

        if current["quality"] < 0.7:
            # Educational retry policy: low quality -> tune and rerun.
            new_quality = round(min(0.98, current["quality"] + 0.17), 3)
            retries.append(
                {
                    "task": current["task"],
                    "reason": "quality_below_threshold",
                    "old_quality": current["quality"],
                    "new_quality": new_quality,
                }
            )
            current["quality"] = new_quality
            current["success"] = True
            current["runtime_minutes"] += 8

        fixed_runs.append(current)

    avg_quality = float(np.mean([r["quality"] for r in fixed_runs]))
    return {
        "retries": retries,
        "post_reflection_runs": fixed_runs,
        "average_quality_after_reflection": round(avg_quality, 3),
        "fatal_errors": 0,
    }


# ------------------------------
# 4) Finalizing stage
# ------------------------------

def finalize_report(tasks: List[Task], reflection: Dict) -> Dict:
    total_runtime = sum(r["runtime_minutes"] for r in reflection["post_reflection_runs"])
    avg_quality = reflection["average_quality_after_reflection"]

    return {
        "artifacts": [
            "clean_dataset.parquet",
            "model.pkl",
            "evaluation_report.json",
            "model_card.md",
        ],
        "human_handoff": {
            "owner": "mlops_lead",
            "approval_required": True,
        },
        "stakeholders_notified": ["data_science", "platform_team", "product_manager"],
        "final_metrics": {
            "total_runtime_minutes": total_runtime,
            "average_quality": avg_quality,
            "tasks_completed": len(tasks),
        },
    }


# ------------------------------
# Visualizations
# ------------------------------

def plot_pipeline_gantt(reflection: Dict) -> None:
    runs = reflection["post_reflection_runs"]
    labels = [r["task"] for r in runs]
    durations = [r["runtime_minutes"] for r in runs]

    plt.figure(figsize=(10, 4.8))
    left = np.cumsum([0] + durations[:-1])
    y = np.arange(len(labels))
    plt.barh(y, durations, left=left, color="#1f77b4")
    plt.yticks(y, labels)
    plt.xlabel("Cumulative Runtime (minutes)")
    plt.title("MLOps Agent Pipeline Runtime (After Reflection)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "page135_140_pipeline_runtime.png", dpi=160)
    plt.close()


def plot_quality_before_after(execution: Dict, reflection: Dict) -> None:
    before = [r["quality"] for r in execution["runs"]]
    after = [r["quality"] for r in reflection["post_reflection_runs"]]
    labels = [r["task"] for r in execution["runs"]]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, before, width=width, label="Before reflection", color="#ff7f0e")
    plt.bar(x + width / 2, after, width=width, label="After reflection", color="#2ca02c")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Quality Score")
    plt.title("Self-Correction Impact by Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "page135_140_reflection_impact.png", dpi=160)
    plt.close()


def plot_framework_comparison() -> None:
    frameworks = ["LangGraph", "CrewAI", "AutoGen"]
    workflow_control = [9, 7, 8]
    setup_speed = [6, 9, 7]
    multi_agent = [8, 8, 9]

    x = np.arange(len(frameworks))
    width = 0.25

    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width, workflow_control, width=width, label="Workflow control", color="#9467bd")
    plt.bar(x, setup_speed, width=width, label="Setup speed", color="#17becf")
    plt.bar(x + width, multi_agent, width=width, label="Multi-agent support", color="#8c564b")
    plt.xticks(x, frameworks)
    plt.ylim(0, 10)
    plt.ylabel("Relative Score (0-10)")
    plt.title("Agent Framework Snapshot (Slides 137-140)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "page135_140_frameworks_comparison.png", dpi=160)
    plt.close()


# ------------------------------
# Main flow
# ------------------------------

def run_demo() -> StageLog:
    log = StageLog()

    tasks = build_task_graph()
    log.planning = summarize_plan(tasks)

    log.execution = simulate_tool_execution(tasks)
    log.reflection = reflect_and_retry(log.execution)
    log.finalizing = finalize_report(tasks, log.reflection)

    plot_pipeline_gantt(log.reflection)
    plot_quality_before_after(log.execution, log.reflection)
    plot_framework_comparison()

    return log


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_log = run_demo()

    output_json = OUTPUT_DIR / "page135_140_run_output.json"
    output_txt = OUTPUT_DIR / "page135_140_run_output.txt"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(stage_log.__dict__, f, indent=2)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write(json.dumps(stage_log.__dict__, indent=2))

    print("Saved:")
    print("- page135_140/page135_140_pipeline_runtime.png")
    print("- page135_140/page135_140_reflection_impact.png")
    print("- page135_140/page135_140_frameworks_comparison.png")
    print("- page135_140/page135_140_run_output.json")
    print("- page135_140/page135_140_run_output.txt")
