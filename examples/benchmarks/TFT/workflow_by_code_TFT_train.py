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
import libs.utils as utils


if __name__ == "__main__":
    # use default data
    GetData().qlib_data(target_dir=utils.PROVIDER_URI, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=utils.PROVIDER_URI, region=REG_CN)

    R.set_uri("./" + utils.MLRUNS_PATH)

    # model initialization
    dataset = init_instance_by_config(utils.TASK_CONFIG["dataset"])
    model = init_instance_by_config(utils.TASK_CONFIG["model"])

    # start exp to train model
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(utils.TASK_CONFIG))
        recorder = R.get_recorder()
        local_dir = recorder.get_local_dir()
        base_dir = utils.get_record_base_dir(local_dir, utils.MLRUNS_PATH)
        model.fit(dataset, base_dir, utils.MLRUNS_PATH)
        R.save_objects(trained_model=model)

        print("workflow_by_code_TFT_train recorder_id: " + recorder.id)
