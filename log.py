from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


class Log:
    """Inspect the metrics.csv file produced by Lightning's CSVLogger."""

    DEFAULT_METRICS = ("normal_loss", "lr")

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()

    def read_dataframe(self) -> pd.DataFrame:
        if not self.path.is_file():
            raise FileNotFoundError(f"Metrics CSV does not exist: {self.path}")
        if self.path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file: {self.path}")

        dataframe = pd.read_csv(self.path)
        if dataframe.empty:
            raise ValueError(f"Metrics CSV is empty: {self.path}")
        return dataframe

    def show_plot(
        self,
        metrics: Optional[Sequence[str]] = None,
        x_column: Optional[str] = None,
    ) -> None:
        dataframe = self.read_dataframe()
        selected_metrics = tuple(metrics or self.DEFAULT_METRICS)
        if not selected_metrics:
            raise ValueError("At least one metric must be selected")

        missing_metrics = [
            metric for metric in selected_metrics if metric not in dataframe.columns
        ]
        if missing_metrics:
            raise ValueError(
                f"Metrics CSV is missing columns {missing_metrics}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        if x_column is None:
            x_column = next(
                (
                    candidate
                    for candidate in ("step", "epoch")
                    if candidate in dataframe.columns
                ),
                None,
            )
        if x_column is not None and x_column not in dataframe.columns:
            raise ValueError(
                f"Metrics CSV is missing x-axis column {x_column!r}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        import matplotlib.pyplot as plt

        x_values = (
            dataframe[x_column]
            if x_column is not None
            else pd.Series(
                range(1, len(dataframe) + 1),
                index=dataframe.index,
            )
        )
        for metric in selected_metrics:
            valid_rows = dataframe[metric].notna()
            plt.plot(
                x_values[valid_rows],
                dataframe.loc[valid_rows, metric],
                marker="o",
                label=metric,
            )
        plt.xlabel(x_column or "record")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_dataframe(self) -> pd.DataFrame:
        dataframe = self.read_dataframe()
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
        ):
            print(dataframe)
        return dataframe
