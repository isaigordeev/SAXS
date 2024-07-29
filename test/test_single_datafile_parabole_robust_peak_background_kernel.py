import unittest
from saxs.gaussian_processing.manager import Manager
from saxs.gaussian_processing.peak.default_kernel import DefaultPeakKernel
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.phase.default_kernel import DefaultPhaseKernel
from saxs.gaussian_processing.phase.primitive_kernel import PrimitivePhaseKernel
from saxs.gaussian_processing.phase.phase_application import PhaseApplication

from saxs.gaussian_processing.peak.prominence_kernel import ProminencePeakKernel
from saxs.gaussian_processing.peak.parabole_kernel import ParabolePeakKernel, RobustParabolePeakKernelWithBackground


class SingleDatafileLabTest(unittest.TestCase):

    def setUp(self):
        self.application = Manager(peak_data_path="test_processing_data/075773_treated_xye.csv" , peak_kernel=RobustParabolePeakKernelWithBackground, phase_kernel=DefaultPhaseKernel)
        self.application()
        self.expected_peaks = 7

    def test_parabole_peak_kernel(self):
        peak_classificator = self.application.peak_application_instance.peak_classificator
        if isinstance(peak_classificator, RobustParabolePeakKernelWithBackground):
            self.assertEqual(len(peak_classificator.peaks), self.expected_peaks)



