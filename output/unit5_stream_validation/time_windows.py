"""Compare timestamp windows with frame windows on validation signers only."""
import hashlib
import json

from replay import ROOT, OUT, rows, targets, replay, summarize

experiments = {}
for fps in (8, 6, 4):
    for seconds in (None, 1.0, 1.5):
        name = f"{'baseline' if seconds is None else str(seconds)+'s'}_{fps}fps"
        results = [replay(row, target, fps, window_seconds=seconds)
                   for row in rows for target in targets]
        summary = summarize(results)
        # Two driving sentences use the word sequence evaluator.
        driving = [row for row in results if row["target"] in ("SEN0009", "SEN0011")]
        summary["driving"] = summarize(driving)
        experiments[name] = dict(summary=summary, results=results)
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
report = dict(
    source_fps_assumption=30,
    changed_scope=["SEN0009", "SEN0011"],
    window_endpoint="inclusive; readiness tolerance 1/30 second for rounded source frames",
    checkpoint_sha256={kind:hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
                       for kind,file in (("sentence","unit5.pt"),("word","unit5_words.pt"))},
    experiments=experiments,
)
(OUT / "time_window_results.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)+"\n")
