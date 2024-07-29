import unittest
from saxs.gaussian_processing.manager import Manager
from saxs.gaussian_processing.peak.default_kernel import DefaultPeakKernel
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.phase.default_kernel import DefaultPhaseKernel
from saxs.gaussian_processing.phase.primitive_kernel import PrimitivePhaseKernel
from saxs.gaussian_processing.phase.phase_application import PhaseApplication

from saxs.gaussian_processing.peak.prominence_kernel import ProminencePeakKernel
from saxs.gaussian_processing.peak.parabole_kernel import ParabolePeakKernel, RobustParabolePeakKernel


class SingleDatafileLabTest(unittest.TestCase):

    def setUp(self):
        self.application = Manager(peak_data_path="test_processing_data/" , peak_kernel=RobustParabolePeakKernel, phase_kernel=DefaultPhaseKernel)
        self.application()
        self.expected_peaks = [6,6,6,6]

    def test_parabole_peak_kernel(self):
        peak_numbers_by_sample = self.application.peak_application_instance.peak_numbers_by_sample
        self.assertEqual(len(self.expected_peaks), len(peak_numbers_by_sample))
        for i, number_of_peaks in enumerate(peak_numbers_by_sample):
            self.assertEqual(number_of_peaks, self.expected_peaks[i])



