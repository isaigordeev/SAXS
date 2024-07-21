import json
import time

from saxs.gaussian_processing.manager import DirectoryManager
from saxs.gaussian_processing.phase.custom_phase_classification import *
from saxs.saxs_model.phase_prediction import prediction_from_csv, prediction_from_npy
import torch


class PipelineAbstract:  # TODO TYPING CLASS
    __slots__ = (
        "data_path",
        "model",
        "peak_kernel",
        "phase_kernel",
        "model_path",
        "model",
        "is_model_prediction",
        "processing_manager"
    )


class Pipeline(PipelineAbstract):
    def __init__(self, data_path, model_path, peak_kernel, phase_kernel):
        self.model = None
        self.processing_manager = None
        self.data_path = data_path
        self.model_path = model_path
        self.peak_kernel = peak_kernel
        self.phase_kernel = phase_kernel

        self.is_model_prediction = True if self.model_path is not None else False
        self.is_processing_prediction = True if self.peak_kernel is not None \
                                                and self.phase_kernel is not None else False

        self.setting()

    def __call__(self, *args, **kwargs):
        self.prediction()

    def setting(self):
        if self.is_model_prediction:
            self.load_model()
        if self.is_processing_prediction:
            self.processing_manager = DirectoryManager(self.data_path, self.peak_kernel, self.phase_kernel)

    def load_model(self):
        self.model = torch.load(self.model_path)

    def model_prediction(self):
        filename, ext = os.path.split(self.data_path)
        if ext == ".csv":
            prediction_from_csv(self.model, self.data_path)  # TODO prediction writes in json
        if ext == ".npy":
            prediction_from_npy(self.model, self.data_path)  # TODO prediction writes in json

    def processing_prediction(self):
        self.processing_manager()

    def prediction(self):
        if self.is_model_prediction:
            self.model_prediction()
        if self.is_processing_prediction:
            self.processing_prediction()
