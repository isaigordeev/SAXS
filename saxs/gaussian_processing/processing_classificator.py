import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime

from .settings_processing import ANALYSE_DIR, ANALYSE_DIR_SESSIONS_RESULTS, ANALYSE_DIR_SESSIONS


@dataclass
class AbstractApplication(ABC):
    current_root_session = datetime.now()
    executing_path = os.getcwd()

    __slots__ = (
        'current_session',
        'current_date_session',
        'current_time',
    )

    def __init__(self, current_session=None):
        if current_session is not None:
            self.current_session = current_session
        else:
            self.current_session = AbstractApplication.current_root_session

        current_date = str(self.current_session.today().date())
        self.current_date_session = "{}/".format(current_date)
        self.current_time = self.current_session.strftime("%H:%M:%S")


class Application(AbstractApplication):
    _results_dir = None
    _result_plots_dir = None
    _total_results_dir_ = None
    _current_results_dir_session = None
    _default_peak_data_path = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def __init__(self, current_session=None, custom_output_directory=None):
        super().__init__(current_session)

        self.custom_output_directory = custom_output_directory
        self.set_output_directories()

        Application._default_peak_data_path = os.path.join(self._current_results_dir_session,
                                               '{}.json'.format(self.current_time))

    def set_output_directories(self):

        if self.custom_output_directory is not None:
            Application._results_dir = os.path.join(self.custom_output_directory, ANALYSE_DIR)
        else:
            Application._results_dir = os.path.join(self.executing_path, ANALYSE_DIR)

        if Application._results_dir is None:
            raise NotADirectoryError("Root output directory error")


        Application._result_plots_dir = os.path.join(Application._results_dir, ANALYSE_DIR_SESSIONS)
        # print(Application._result_plots_dir)
        Application._total_results_dir_ = os.path.join(Application._results_dir, ANALYSE_DIR_SESSIONS_RESULTS)
        Application._current_results_dir_session = os.path.join(Application._total_results_dir_,
                                                                super().current_date_session)

        if not os.path.exists(Application._results_dir):
            os.mkdir(Application._results_dir)
        if not os.path.exists(Application._result_plots_dir):
            os.mkdir(Application._result_plots_dir)
        if not os.path.exists(Application._total_results_dir_):
            os.mkdir(Application._total_results_dir_)
        if not os.path.exists(Application._current_results_dir_session):
            os.mkdir(Application._current_results_dir_session)




@dataclass
class ApplicationClassificator(Application): #TODO another abstract class for classificator application
    # _data_directory = None

    def __init__(self, data_directory):
        assert data_directory is not None
        super().__init__(None, None)

        self.data = {}
        self.data_directory = data_directory
        self.kernel = None
