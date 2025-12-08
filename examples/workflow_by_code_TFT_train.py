#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK

# sys:
#     rel_path: .
# qlib_init:
#     provider_uri: "~/.qlib/qlib_data/cn_data"
#     region: cn
# market: & market csi300
# benchmark: & benchmark SH000300
# data_handler_config: & data_handler_config
#    start_time: 2008-01-01
#     end_time: 2020-08-01
#     fit_start_time: 2008-01-01
#     fit_end_time: 2014-12-31
#     instruments: *market
# port_analysis_config: & port_analysis_config
#    strategy:
#         class:
#             TopkDropoutStrategy
#         module_path: qlib.contrib.strategy
#         kwargs:
#             signal: < PRED >
#             topk: 50
#             n_drop: 5
#     backtest:
#         start_time: 2017-01-01
#         end_time: 2020-08-01
#         account: 100000000
#         benchmark: *benchmark
#         exchange_kwargs:
#             limit_threshold: 0.095
#             deal_price: close
#             open_cost: 0.0005
#             close_cost: 0.0015
#             min_cost: 5
# task:
#     model:
#         class:
#             TFTModel
#         module_path: tft
#     dataset:
#         class:
#             DatasetH
#         module_path: qlib.data.dataset
#         kwargs:
#             handler:
#                 class:
#                     Alpha158
#                 module_path: qlib.contrib.data.handler
#                 kwargs: *data_handler_config
#             segments:
#                 train: [2008-01-01, 2014-12-31]
#                 valid: [2015-01-01, 2016-12-31]
#                 test: [2017-01-01, 2020-08-01]
#     record:
#         - class: SignalRecord
#          module_path: qlib.workflow.record_temp
#           kwargs:
#             model: < MODEL>
#             dataset: < DATASET>
#         - class: SigAnaRecord
#          module_path: qlib.workflow.record_temp
#           kwargs:
#             ana_long_short: False
#             ann_scaler: 252
#         - class: PortAnaRecord
#          module_path: qlib.workflow.record_temp
#           kwargs:
#             config: *port_analysis_config

#    "learn_processors": [{
#        "class": "DropnaLabel"
#    }, {
#        "class": "CSRankNorm",
#        "kwargs": {
#            "fields_group": "label"
#        }
#    }],
#    "label": ["Ref($close, -2) / Ref($close, -1) - 1"]

if __name__ == "__main__":
    # use default data
    provider_uri = "./qlib_data/qlib_data_hs300_5min"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300s19_22"
    benchmark = "SH000300"

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2020-09-15 00:00:00",
        "end_time": "2021-01-18 16:00:00",
        "fit_start_time": "2020-09-15 00:00:00",
        "fit_end_time": "2020-11-15 16:00:00",
        "instruments": market,
        "freq": "5min",
        "infer_processors": [{
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }
        }, {
            "class": "Fillna",
            "kwargs": {
                "fields_group": "feature"
            }
        }],
        "learn_processors": [{
            "class": "DropnaLabel"
        }, {
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }
        }]
    }

    task = {
        "model": {
            "class": "TFTModel",
            "module_path": "benchmarks.TFT.tft",
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2020-09-15 00:00:00", "2020-11-15 16:00:00"),
                    "valid": ("2020-11-15 16:00:00", "2020-11-30 16:00:00"),
                    "test": ("2020-12-01 00:00:00", "2021-01-18 16:00:00"),
                },
                "seq_len": 78
            },
        },
    }

    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        print("workflow_by_code_TFT_train recorder_id: " + rid)
