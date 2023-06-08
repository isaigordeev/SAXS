import pandas as pd

from processing.peak_classification import *
from processing.phase_classification import *
import os
import json
from datetime import date, datetime
import time

today = date.today()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

current_session = ANALYSE_DIR_SESSIONS + str(today) + '/'
current_session_results = ANALYSE_DIR_SESSIONS_RESULTS + str(today) + '/'

if not os.path.exists(current_session):
    os.mkdir(current_session)
if not os.path.exists(current_session_results):
    os.mkdir(current_session_results)


def get_filenames(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            yield filename


def get_filenames_without_ext(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            name, extension = os.path.splitext(filename)
            yield name


data = {}
files_number = 0
time_start = time.time()

get_breaked = True

class Manager:
    def __init__(self, current_session: str, DATA_DIR=DATA_DIR):
        self.DATA_DIR = DATA_DIR
        self.current_session = current_session
        self.data = {}
        self.files_number = 0

    def atomic_processing(self, filename):
        peaks = Peaks(filename, self.DATA_DIR, current_session=self.current_session)
        peaks.background_reduction()
        peaks.filtering()
        peaks.background_plot()
        peaks.filtering_negative()
        peaks.peak_processing()
        peaks.result_plot()
        self.data[filename] = peaks.gathering()
        self.files_number += 1
        # phase TODO

    def repo_processing(self):
        filenames = get_filenames_without_ext(self.DATA_DIR)
        for filename in filenames:
            self.atomic_processing(filename)

    def print_data(self):
        print(self.data)

    def write_data(self):
        with open(current_session_results + current_time + f'_{self.files_number}.json', 'w') as f:
            json.dump(self.data, f, indent=4, separators=(",", ": "))


manager = Manager(current_session)
manager.repo_processing()
# manager.atomic_processing('075790_treated_xye')
manager.write_data()

