import matplotlib.pyplot as plt
import pandas as pd
from pathlib  import Path

class Log():
    def __init__(self,path: Path):
        self.path = path

    def show_plot(self):
        df = pd.read_csv(self.path)
        prediction_count = len(df["predictions"])
        target_count = len(df["targets"])

        print("当前文件：", __file__)
        print("预测数量：", prediction_count)
        print("目标数量：", target_count)
        print("是否不相等：", prediction_count != target_count)
        if prediction_count != target_count:
            raise ValueError(
                f"预测值数量为 {prediction_count}\n"
                f"目标值数量为 {target_count}"
            )
        indexes = range(1, len(df["targets"]) + 1)
        plt.plot(indexes, df["predictions"], color='blue', marker='o')
        plt.plot(indexes, df["targets"], color='red', marker='x')
        plt.legend()
        plt.grid(True)
        plt.show()

    def show_dataframe(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        csv = pd.read_csv(self.path)
        print(csv)

