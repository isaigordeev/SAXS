import unittest
from saxs.gaussian_processing.manager import Manager
from saxs.gaussian_processing.phase.phase_kernel.default_kernel import DefaultPhaseKernel

from saxs.gaussian_processing.peak.peak_kernel.robust_parabole_kernel import RobustParabolePeakKernelWithBackground


class SingleDatafileLabTest(unittest.TestCase):

    def setUp(self):
        self.application = Manager(peak_data_path="test_processing_data/075776_treated_xye.csv" , peak_kernel=RobustParabolePeakKernelWithBackground, phase_kernel=DefaultPhaseKernel)
        self.application()
        self.expected_peaks = 7

    def test_parabole_peak_kernel(self):
        peak_classificator = self.application.peak_application_instance.peak_classificator
        if isinstance(peak_classificator, RobustParabolePeakKernelWithBackground):
            self.assertEqual(len(peak_classificator.peaks), self.expected_peaks)



