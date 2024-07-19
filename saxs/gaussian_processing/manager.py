import time
from typing import Any

from .application import ApplicationManager
from .peak.abstract_kernel import AbstractPeakKernel
from .peak.peak_application import PeakApplication
from .phase.phase_application import PhaseApplication
from .phase.phase_classificator import AbstractPhaseKernel

time_start1 = time.time()

import os

from datetime import datetime

time_start2 = time.time()

now = datetime.now()

today = now.today().date()


class Manager(ApplicationManager):
    def __init__(self,
                 peak_data_directory='test_processing_data/',
                 phase_data_directory=None,
                 peak_kernel: AbstractPeakKernel = None,
                 phase_kernel: AbstractPhaseKernel = None,
                 current_session=None) -> None:
        super().__init__(current_session=current_session,
                         peak_kernel=peak_kernel,
                         phase_kernel=phase_kernel)

        self.peak_data_directory = os.path.join(self.executing_path, peak_data_directory)
        self.phase_data_directory = self._default_peak_data_path if phase_data_directory is None else phase_data_directory

    def __call__(self):
        self.peak_application_instance = PeakApplication(self.peak_data_directory, self.peak_kernel)
        self.peak_application_instance.peak_classification_run()

        self.phase_application_instance = PhaseApplication(self.phase_data_directory, self.phase_kernel)
        self.phase_application_instance.phase_classification_run()
