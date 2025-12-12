# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import os
import datetime as dte


from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R


# To register new datasets, please add them here.
ALLOW_DATASET = ["Alpha158", "Alpha360"]
# To register new datasets, please add their configurations here.
DATASET_SETTING = {
    "Alpha158": {
        "feature_col": [
            "RESI5",
            "WVMA5",
            "RSQR5",
            "KLEN",
            "RSQR10",
            "CORR5",
            "CORD5",
            "CORR10",
            "ROC60",
            "RESI10",
            "VSTD5",
            "RSQR60",
            "CORR60",
            "WVMA60",
            "STD5",
            "RSQR20",
            "CORD60",
            "CORD10",
            "CORR20",
            "KLOW",
        ],
        "label_col": "LABEL0",
    },
    "Alpha360": {
        "feature_col": [
            "HIGH0",
            "LOW0",
            "OPEN0",
            "CLOSE1",
            "HIGH1",
            "VOLUME1",
            "LOW1",
            "VOLUME3",
            "OPEN1",
            "VOLUME4",
            "CLOSE2",
            "CLOSE4",
            "VOLUME5",
            "LOW2",
            "CLOSE3",
            "VOLUME2",
            "HIGH2",
            "LOW4",
            "VOLUME8",
            "VOLUME11",
        ],
        "label_col": "LABEL0",
    },
}


def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    return data_df[[col_shift]].groupby("instrument", group_keys=False).apply(lambda df: df.shift(shifts))


def fill_test_na(test_df):
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        test_df_res.loc[:, feature_cols].groupby("datetime", group_keys=False).apply(lambda df: df.fillna(df.mean()))
    )
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


def process_qlib_data(df, dataset, fillna=False):
    """Prepare data to fit the TFT model.

    Args:
      df: Original DataFrame.
      fillna: Whether to fill the data with the mean values.

    Returns:
      Transformed DataFrame.

    """
    # Several features selected manually
    feature_col = DATASET_SETTING[dataset]["feature_col"]
    label_col = [DATASET_SETTING[dataset]["label_col"]]
    temp_df = df.loc[:, feature_col + label_col]
    if fillna:
        temp_df = fill_test_na(temp_df)
    temp_df = temp_df.swaplevel()
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)
    dates = pd.to_datetime(temp_df.index)
    temp_df["date"] = dates
    temp_df["day_of_week"] = dates.dayofweek
    temp_df["month"] = dates.month
    temp_df["year"] = dates.year
    temp_df["const"] = 0
    return temp_df


def process_predicted(df, col_name):
    """Transform the TFT predicted data into Qlib format.

    Args:
      df: Original DataFrame.
      col_name: New column name.

    Returns:
      Transformed DataFrame.

    """
    df_res = df.copy()
    df_res = df_res.rename(columns={"forecast_time": "datetime", "identifier": "instrument", "t+4": col_name})
    df_res = df_res.set_index(["datetime", "instrument"]).sort_index()
    df_res = df_res[[col_name]]
    return df_res


def format_score(forecast_df, col_name="pred", label_shift=5):
    pred = process_predicted(forecast_df, col_name=col_name)
    pred = get_shifted_label(pred, shifts=-label_shift, col_shift=col_name)
    pred = pred.dropna()[col_name]
    return pred


def transform_df(df, col_name="LABEL0"):
    df_res = df["feature"]
    df_res[col_name] = df["label"]
    return df_res


class TFTModel(ModelFT):
    """TFT Model"""

    def __init__(self, **kwargs):
        self.model = None
        self.params = {"DATASET": "Alpha158", "label_shift": 5}
        self.params.update(kwargs)

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        return transform_df(df_train), transform_df(df_valid)

    def _filter_instruments(self, date: pd.DataFrame):
        # 过滤小于time_steps的数据实体
        # time_steps = int(params['total_time_steps'])
        # for id, sliced in train.groupby("instrument"):
        #     if (len(sliced) < time_steps):
        #         print(f"{id},")

        # for id, sliced in valid.groupby("instrument"):
        #     if (len(sliced) < time_steps):
        #         print(f"{id},")

        # counts = train["instrument"].value_counts()
        # valid_inst = counts[counts >= time_steps].index
        # train = train[train["instrument"].isin(valid_inst)]

        # counts = valid["instrument"].value_counts()
        # valid_inst = counts[counts >= time_steps].index
        # valid = valid[valid["instrument"].isin(valid_inst)]

        exclude_ids = ["SH600038", "SH600373", "SH600398", "SH600485", "SH600570", "SH600578", "SH603288", "SZ002153",
                       "SZ300015", "SZ300017", "SZ300024", "SZ300027", "SZ300058", "SZ300070", "SZ300124", "SZ300133",
                       "SZ300146", "SZ300251", "SH600297", "SH600482", "SH600754", "SH601127", "SH601155", "SH601611",
                       "SH601877", "SZ000008", "SZ000555", "SZ000627", "SZ000671", "SZ000718", "SZ000938", "SZ002049",
                       "SZ002074", "SZ002085", "SZ002131", "SZ002174", "SZ002299", "SZ002426", "SZ002466", "SZ002714",
                       "SZ002797", "SZ300033", "SZ300072", "SZ300182"]

        return date[~date["instrument"].isin(exclude_ids)]

    def _make_model_path(self, base_dir: str, experiment_id: str = None, recorder_id: str = None):
        exp_info = R.get_exp().info
        if experiment_id is None:
            experiment_id = exp_info["id"]
        if recorder_id is None:
            recorder_id = exp_info["active_recorder"]
        model_path = os.path.join(base_dir, self.model_folder, "models", str(experiment_id), str(recorder_id))
        return model_path

    def fit(self, dataset: DatasetH, BASE_DIR, MODEL_FOLDER="qlib_tft_model", USE_GPU_ID=0, FILTER_INSTRUMENTS=False):
        DATASET = self.params["DATASET"]
        LABEL_SHIFT = self.params["label_shift"]
        LABEL_COL = DATASET_SETTING[DATASET]["label_col"]

        if DATASET not in ALLOW_DATASET:
            raise AssertionError("The dataset is not supported, please make a new formatter to fit this dataset")

        dtrain, dvalid = self._prepare_data(dataset)
        dtrain.loc[:, LABEL_COL] = get_shifted_label(dtrain, shifts=LABEL_SHIFT, col_shift=LABEL_COL)
        dvalid.loc[:, LABEL_COL] = get_shifted_label(dvalid, shifts=LABEL_SHIFT, col_shift=LABEL_COL)

        train = process_qlib_data(dtrain, DATASET, fillna=True).dropna()
        valid = process_qlib_data(dvalid, DATASET, fillna=True).dropna()

        # pd.set_option('display.min_rows', 300)  # 最多显示50行
        ExperimentConfig = expt_settings.configs.ExperimentConfig
        config = ExperimentConfig(DATASET)
        self.data_formatter = config.make_data_formatter()
        self.model_folder = MODEL_FOLDER
        self.gpu_id = USE_GPU_ID
        self.label_shift = LABEL_SHIFT
        self.expt_name = DATASET
        self.label_col = LABEL_COL
        self.filter_instruments = FILTER_INSTRUMENTS

        use_gpu = (True, self.gpu_id)
        # ===========================Training Process===========================
        ModelClass = libs.tft_model.TemporalFusionTransformer
        if not isinstance(self.data_formatter, data_formatters.base.GenericDataFormatter):
            raise ValueError(
                "Data formatters should inherit from"
                + "AbstractDataFormatter! Type={}".format(type(self.data_formatter))
            )

        if use_gpu[0]:
            utils.set_tf_device(tf_device="gpu")
        else:
            utils.set_tf_device(tf_device="cpu")

        self.data_formatter.set_scalers(train)

        train = self.data_formatter.transform_inputs(train)
        valid = self.data_formatter.transform_inputs(valid)

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()

        params = {**params, **fixed_params}

        model_folder = self._make_model_path(BASE_DIR)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        params["model_folder"] = model_folder

        if self.filter_instruments:
            train = self._filter_instruments(train)
            valid = self._filter_instruments(valid)

        print("*** Begin training ***")
        self.model = ModelClass(params, use_cudnn=use_gpu[0])
        self.model.fit(train_df=train, valid_df=valid)
        self.model.save(model_folder)

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[col for col in data.columns if col not in {"forecast_time", "identifier"}]]

        # p50_loss = utils.numpy_normalised_quantile_loss(
        #    extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        #    0.5)
        # p90_loss = utils.numpy_normalised_quantile_loss(
        #    extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        #    0.9)
        # tf.keras.backend.set_session(default_keras_session)
        print("Training completed at {}.".format(dte.datetime.now()))
        # ===========================Training Process===========================

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, self.label_col] = get_shifted_label(d_test, shifts=self.label_shift, col_shift=self.label_col)
        test = process_qlib_data(d_test, self.expt_name, fillna=True).dropna()
        test = self.data_formatter.transform_inputs(test)

        if self.filter_instruments:
            test = self._filter_instruments(test)

        # # ===========================Predicting Process===========================

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}

        print("*** Begin predicting ***")

        output_map = self.model.predict(test, return_targets=True)

        # targets = self.data_formatter.format_predictions(output_map["targets"])
        # p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
        # p90_forecast = self.data_formatter.format_predictions(output_map["p90"])

        targets = output_map["targets"]
        p50_forecast = output_map["p50"]
        p90_forecast = output_map["p90"]

        predict50 = format_score(p50_forecast, "pred", 1)
        predict90 = format_score(p90_forecast, "pred", 1)
        predict = (predict50 + predict90) / 2  # self.label_shift

        # ===========================Predicting Process===========================
        return predict

    def finetune(self, dataset: DatasetH):
        """
        finetune model
        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        """
        pass

    def load(self, base_dir: str, experiment_id: str = None, recorder_id: str = None):
        self.data_formatter.show_params()

        use_gpu = (True, self.gpu_id)
        # ===========================Training Process===========================
        ModelClass = libs.tft_model.TemporalFusionTransformer
        if not isinstance(self.data_formatter, data_formatters.base.GenericDataFormatter):
            raise ValueError(
                "Data formatters should inherit from"
                + "AbstractDataFormatter! Type={}".format(type(self.data_formatter))
            )

        if use_gpu[0]:
            utils.set_tf_device(tf_device="gpu")
        else:
            utils.set_tf_device(tf_device="cpu")

        # Sets up default params
        fixed_params = self.data_formatter.get_experiment_params()
        params = self.data_formatter.get_default_model_params()

        params = {**params, **fixed_params}
        model_path = self._make_model_path(base_dir, experiment_id, recorder_id)
        params["model_folder"] = model_path

        self.model = ModelClass(params, use_cudnn=use_gpu[0])
        self.model.load(model_path, True)

    def to_pickle(self, path: Union[Path, str]):
        """
        Tensorflow model can't be dumped directly.
        So the data should be save separately

        **TODO**: Please implement the function to load the files

        Parameters
        ----------
        path : Union[Path, str]
            the target path to be dumped
        """
        # FIXME: implementing saving tensorflow models
        # save tensorflow model
        # path = Path(path)
        # path.mkdir(parents=True)
        # self.model.save(path)

        # save qlib model wrapper
        drop_attrs = ["model"]
        orig_attr = {}
        for attr in drop_attrs:
            orig_attr[attr] = getattr(self, attr)
            setattr(self, attr, None)
        super(TFTModel, self).to_pickle(path)
        for attr in drop_attrs:
            setattr(self, attr, orig_attr[attr])
