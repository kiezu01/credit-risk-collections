"""Command line entry point: ``uv run crc <command>``.

crc sample   generate a synthetic sample (for tests / a quick try-out)
crc build    build the DuckDB warehouse with dbt (models + data tests)
crc train    train the models and write reports/
crc all      build + train
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import duckdb

from . import challenger, collections_model, experiment, reporting, sample_data, scorecard
from .config import ROOT, get_paths

# Daily volume of new DPD 5 entries, used to turn A/B sample sizes into a test duration.
# Home Credit has no calendar dates, so this is an assumption: use the real portfolio's volume.
EVENTS_PER_DAY = 2000


def build() -> None:
    paths = get_paths()
    env = {**os.environ, "CRC_RAW_DIR": str(paths.raw_dir), "CRC_DUCKDB_PATH": str(paths.duckdb)}
    cmd = ["dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def train() -> dict:
    paths = get_paths()
    with duckdb.connect(str(paths.duckdb), read_only=True) as con:
        print("[1/4] Application scorecard ...", flush=True)
        df = scorecard.load_features(con)
        sc = scorecard.train(df)
        print("      test", sc.metrics["test"], flush=True)

        print("[2/4] LightGBM challenger ...", flush=True)
        train_df = df[df[scorecard.ID].isin(sc.train[scorecard.ID])]
        test_df = df.set_index(scorecard.ID).loc[sc.test[scorecard.ID]].reset_index()
        ch = challenger.train(train_df, test_df, sc.features, scorecard.TARGET)
        print("      test", ch.metrics["test"], flush=True)

        print("[3/4] Collections roll-rate model + prioritisation ...", flush=True)
        col = collections_model.run(con)
        print("      oot", col.metrics["oot"], flush=True)

    print("[4/4] A/B test design + reports ...", flush=True)
    worked = col.oot_scored.assign(
        rank=(1 - col.oot_scored["p_cure"]) * col.oot_scored["shortfall_amt"]
    ).nlargest(int(0.3 * len(col.oot_scored)), "rank")
    # baseline = roll rate of the accounts the model would put in front of collectors
    exp_table = experiment.design_table(1 - float(worked["is_cured"].mean()), EVENTS_PER_DAY)

    results = reporting.write_all(
        paths, sc, ch, col, exp_table, collections_model.STRATEGIES, EVENTS_PER_DAY
    )
    print(json.dumps(results, indent=2, default=float))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="crc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["sample", "build", "train", "all"])
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "sample"), help="output folder for `sample`"
    )
    args = parser.parse_args(argv)

    if args.command == "sample":
        print(f"Sample written to {sample_data.generate(args.out)}")
    elif args.command == "build":
        build()
    elif args.command == "train":
        train()
    else:
        build()
        train()


if __name__ == "__main__":
    sys.exit(main())
