#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.contrib.report import analysis_position

if __name__ == "__main__":
    # use default data
    provider_uri = "./qlib_data_gru/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    R.set_uri("./mlruns_gru")

    rid = "dbc7e737541342f49b342f3a276ba004"

    recorder = R.get_recorder(recorder_id=rid, experiment_name="backtest_analysis")

    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")

    df = pd.concat([pred_df, label_df], axis=1).dropna()
    pred, label = df.iloc[:, 0], df.iloc[:, 1]

    MSE = ((pred - label) ** 2).mean()
    MAE = (pred - label).abs().mean()

    print("MSE:", MSE)
    print("MAE:", MAE)

    # print(pred_df)
    # print("|||||||||||||||||||||||||||||")
    # print(label_df)
    # report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    # positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    # analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    # analysis_position.report_graph(report_normal_df, True)
