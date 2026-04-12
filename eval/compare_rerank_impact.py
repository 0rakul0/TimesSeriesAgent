from __future__ import annotations

import io
import math
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "experiment_logs"


def _read_git_csv(path: str) -> pd.DataFrame:
    content = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True)
    return pd.read_csv(io.StringIO(content))


def _rmse(df: pd.DataFrame, pred_col: str) -> float:
    return math.sqrt(((df["Real"] - df[pred_col]) ** 2).mean())


def _collect_current() -> pd.DataFrame:
    rows = []
    for path in sorted(LOG_DIR.glob("prediction_details_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        rows.append(
            {
                "Ativo": str(df["Ativo"].iloc[0]),
                "Modelo": str(df["Modelo"].iloc[0]).upper(),
                "RMSE_Hibrido_After": _rmse(df, "Pred_Hibrido"),
                "RMSE_Hibrido_Event_After": _rmse(df[df["Event_Day"] == True], "Pred_Hibrido"),
                "RMSE_Base_Event_After": _rmse(df[df["Event_Day"] == True], "Pred_Base"),
            }
        )
    return pd.DataFrame(rows)


def _collect_before() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "data" / "experiment_logs").glob("prediction_details_*.csv")):
        rel = path.relative_to(ROOT).as_posix()
        df = _read_git_csv(rel)
        rows.append(
            {
                "Ativo": str(df["Ativo"].iloc[0]),
                "Modelo": str(df["Modelo"].iloc[0]).upper(),
                "RMSE_Hibrido_Before": _rmse(df, "Pred_Hibrido"),
                "RMSE_Hibrido_Event_Before": _rmse(df[df["Event_Day"] == True], "Pred_Hibrido"),
                "RMSE_Base_Event_Before": _rmse(df[df["Event_Day"] == True], "Pred_Base"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    before = _collect_before()
    after = _collect_current()
    merged = before.merge(after, on=["Ativo", "Modelo"], how="inner")
    merged["Delta_RMSE_Hibrido"] = merged["RMSE_Hibrido_After"] - merged["RMSE_Hibrido_Before"]
    merged["Delta_RMSE_Event"] = merged["RMSE_Hibrido_Event_After"] - merged["RMSE_Hibrido_Event_Before"]
    merged["Delta_Ganho_Event"] = (
        (merged["RMSE_Base_Event_After"] - merged["RMSE_Hibrido_Event_After"])
        - (merged["RMSE_Base_Event_Before"] - merged["RMSE_Hibrido_Event_Before"])
    )

    cols = [
        "Ativo",
        "Modelo",
        "RMSE_Hibrido_Before",
        "RMSE_Hibrido_After",
        "Delta_RMSE_Hibrido",
        "RMSE_Hibrido_Event_Before",
        "RMSE_Hibrido_Event_After",
        "Delta_RMSE_Event",
        "Delta_Ganho_Event",
    ]
    print(merged[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
