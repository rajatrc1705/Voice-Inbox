import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import agent
from livekit.agents import ErrorEvent
from voice_inbox.evaluation import grade_turn, observe_run
from voice_inbox.repository import VoiceInboxRepository
from voice_inbox.workspace import WorkspaceFiles
from voice_inbox.web import extract_page_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = ROOT / "evals" / "cases.json"
DEFAULT_RESULTS_DIRECTORY = ROOT / "eval_results"


def load_cases(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)
    if not isinstance(cases, list):
        raise ValueError("The eval case file must contain a JSON list.")
    return cases


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def metric_name(check_name: str) -> str:
    if check_name in {"tool_selection", "tool_order"}:
        return "tool_selection"
    if check_name.startswith("tool_") and check_name.endswith("_success"):
        return "tool_execution"
    if check_name.startswith("tool_"):
        return "tool_arguments"
    if check_name == "clarification":
        return "clarification"
    if check_name.startswith("state_"):
        return "application_state"
    return "assistant_response"


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {}
    durations: list[float] = []

    for result in results:
        for turn in result["turns"]:
            durations.append(turn["observation"]["duration"])
            for check in turn["checks"]:
                metric = metric_name(check["name"])
                metric_counts = counts.setdefault(metric, {"passed": 0, "total": 0})
                metric_counts["total"] += 1
                metric_counts["passed"] += int(check["passed"])

    metrics: dict[str, object] = {}
    for name, metric_counts in counts.items():
        metrics[name] = {
            **metric_counts,
            "rate": metric_counts["passed"] / metric_counts["total"],
        }
    metrics["case_success"] = {
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "rate": sum(result["passed"] for result in results) / len(results),
    }
    metrics["average_turn_duration"] = (
        sum(durations) / len(durations) if durations else None
    )
    return metrics


async def run_case(
    case: dict[str, object],
    realtime_model: object,
) -> dict[str, object]:
    now = datetime.fromisoformat(case["now"])
    turn_results: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        repository = VoiceInboxRepository(Path(temporary_directory) / "eval.db")
        repository.initialize()
        workspace_root = Path(temporary_directory) / "workspace"
        workspace_root.mkdir()
        for relative_path, content in case.get("workspace_files", {}).items():
            path = workspace_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        web_page = case.get("web_page")
        web_reader = (
            (lambda url: extract_page_text(web_page["html"], url))
            if web_page else agent.read_webpage
        )
        session = agent.build_session(
            repository,
            realtime_model,
            workspace=WorkspaceFiles(workspace_root),
            page_url=web_page["url"] if web_page else None,
            web_reader=web_reader,
        )
        runtime_errors: list[str] = []

        @session.on("error")
        def record_runtime_error(event: ErrorEvent) -> None:
            runtime_errors.append(str(event.error))

        await session.start(agent=agent.build_agent(now), record=False)

        try:
            for turn_number, turn in enumerate(case["turns"], start=1):
                user_input = turn["user"]
                session.userdata.source_transcript = user_input
                session.userdata.transcript_ready.set()

                started_at = monotonic()
                run_result = await session.run(user_input=user_input)
                duration = monotonic() - started_at
                observation = observe_run(run_result.events, duration)
                if runtime_errors:
                    raise RuntimeError(runtime_errors[-1])
                if not observation.assistant_response and not observation.tool_calls:
                    raise RuntimeError("The agent produced no observable output.")
                checks = grade_turn(turn["expected"], observation, repository)

                turn_results.append(
                    {
                        "turn": turn_number,
                        "user": user_input,
                        "passed": all(check.passed for check in checks),
                        "observation": observation.to_dict(),
                        "checks": [check.to_dict() for check in checks],
                    }
                )
        finally:
            await session.aclose()

    return {
        "id": case["id"],
        "kind": case.get("kind", "regression"),
        "tags": case.get("tags", []),
        "passed": all(turn["passed"] for turn in turn_results),
        "turns": turn_results,
    }


async def run_evaluations(
    cases_path: Path, repetitions: int, case_id: str | None = None
) -> dict[str, object]:
    cases = load_cases(cases_path)
    if case_id is not None:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            raise ValueError(f"Unknown eval case: {case_id}")
    results = []
    realtime_model = agent.build_realtime_model()

    try:
        for repetition in range(1, repetitions + 1):
            for case in cases:
                print(f"Running {case['id']} ({repetition}/{repetitions})...")
                try:
                    result = await run_case(case, realtime_model)
                except Exception as error:
                    result = {
                        "id": case["id"],
                        "kind": case.get("kind", "regression"),
                        "tags": case.get("tags", []),
                        "passed": False,
                        "execution_error": str(error),
                        "turns": [],
                    }
                result["repetition"] = repetition
                results.append(result)
    finally:
        await realtime_model.aclose()

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": current_git_commit(),
        "model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        "cases_path": str(cases_path),
        "repetitions": repetitions,
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "metrics": summarize_results(results),
        "results": results,
    }


def print_summary(report: dict[str, object]) -> None:
    print()
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        if "execution_error" in result:
            status = "ERROR"
        print(f"{status} {result['id']} (run {result['repetition']})")
        if "execution_error" in result:
            print(f"  {result['execution_error']}")
        for turn in result["turns"]:
            for check in turn["checks"]:
                if not check["passed"]:
                    print(
                        f"  turn {turn['turn']} {check['name']}: "
                        f"expected {check['expected']!r}, got {check['actual']!r}"
                    )
    print(f"\nPassed {report['passed']}/{report['total']} case runs")
    print("Metrics:")
    for name, metric in report["metrics"].items():
        if name == "average_turn_duration":
            if metric is not None:
                print(f"  {name}: {metric:.2f}s")
            continue
        print(
            f"  {name}: {metric['passed']}/{metric['total']} "
            f"({metric['rate']:.1%})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Voice Inbox agent evaluations.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_evaluations(args.cases, args.repetitions, args.case))
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_RESULTS_DIRECTORY / f"eval-{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Report: {output_path}")

    regression_failures = [
        result
        for result in report["results"]
        if result["kind"] == "regression" and not result["passed"]
    ]
    if regression_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
