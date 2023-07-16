import json
import os

from .processing_classificator import Application
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.phase.abstr_phase import AbstractPhaseClassificator


class ApplicationManager(Application):
    def __init__(self,
                 current_session=None,
                 peak_classificator: PeakApplication = None,
                 phase_classificator: AbstractPhaseClassificator= None,
                 custom_output_directory=None
                 ) -> None:
        super().__init__(current_session, custom_output_directory)


        self.data = {}
        self.files_number = 0
        self.peak_classificator = peak_classificator
        self.phase_classificator = phase_classificator
        self.custom_output_directory = custom_output_directory

        # self.set_directories()
        self.write_data()  # create json

    def write_data(self):
        write_json_path = os.path.join(self._current_results_dir_session, '{}.json'.format(self.current_time))

        with open(write_json_path, 'w') as f:
            json.dump(self.data, f)

    def point_peak_processing(self, sample):
        pass

    def point_phase_processing(self, sample):
        pass

    def directory_peak_processing(self):
        pass

    def directory_phase_processing(self):
        pass
    def point_processing(self, sample):
        pass

    def directory_processing(self):
        self.directory_peak_processing()
        self.directory_phase_processing()

    def custom_directory_processing(self):
        self.custom_process()

    def custom_process(self):
        pass
