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
from qlib.data.data import D
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


if __name__ == "__main__":
    # use default data
    provider_uri = "./qlib_data_gru/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    R.set_uri("./mlruns_gru")

    market = "all"
    benchmark = "sh600000"

    instruments = D.instruments(market='all')
    print(instruments)

    fields = ['$open', '$close', '$high', '$low', '$volume']
    f = D.features(["sh600000"], fields)
    print(f)

    ###################################
    # train model
    ###################################
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": market,
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
            "class": "CSRankNorm",
            "kwargs": {
                "fields_group": "label"
            },
        }],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
    }

    task = {
        "model": {
            "class": "GRU",
            "module_path": "qlib.contrib.model.pytorch_gru",
            "kwargs": {
                "d_feat": 6,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout":  0.0,
                "n_epochs": 200,
                "lr": 1e-3,
                "early_stop": 20,
                "batch_size": 800,
                "metric": "loss",
                "loss": "mse",
                "GPU": 0
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha360",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2014-12-31"),
                    "valid": ("2015-01-01", "2016-12-31"),
                    "test": ("2017-01-01", "2020-08-01"),
                },
            },
        },
    }

    # model initialization
    # model = init_instance_by_config(task["model"])
    # dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    # with R.start(experiment_name="train_model"):
    #     R.log_params(**flatten_dict(task))
    #     model.fit(dataset)
    #     R.save_objects(trained_model=model)
    #     rid = R.get_recorder().id

    #     print("workflow_by_code_gru_train recorder_id:" + rid)
