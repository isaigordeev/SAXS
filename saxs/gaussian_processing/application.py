import json
import os
from typing import Any

from .processing_classificator import Application
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.phase.phase_application import PhaseApplication


class ApplicationManager(Application):
    def __init__(self,
                 peak_kernel: PeakApplication = None,
                 phase_kernel: PhaseApplication = None,
                 current_session=None,
                 custom_output_directory=None
                 ) -> None:
        super().__init__(current_session, custom_output_directory)

        self.data = {}
        self.files_number = 0
        self.peak_kernel = peak_kernel
        self.phase_kernel = phase_kernel
        self.custom_output_directory = custom_output_directory

        self.peak_classification = True if self.peak_kernel is not None else False
        self.phase_classification = True if self.phase_kernel is not None else False
