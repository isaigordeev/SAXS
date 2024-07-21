from datetime import datetime

from saxs.gaussian_processing.manager import DirectoryManager
from saxs.gaussian_processing.peak.default_kernel import DefaultPeakKernel
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.phase.default_kernel import DefaultPhaseKernel
from saxs.gaussian_processing.phase.primitive_kernel import PrimitivePhaseKernel
from saxs.gaussian_processing.phase.phase_application import PhaseApplication

from saxs.gaussian_processing.peak.prominence_kernel import ProminencePeakKernel
from saxs.gaussian_processing.peak.parabole_kernel import ParabolePeakKernel, RobustParabolePeakKernel
from saxs.gaussian_processing.phase.phase_classificator import AbstractPhaseKernel

# a = ProminenceKernel('/Users/isaigordeev/Desktop/2023/saxs/test_processing_data/075775_treated_xye.csv')

# a = PeakApplication('/Users/isaigordeev/Desktop/2023/saxs/test_processing_data', ParabolePeakKernel)
# a.peak_classification()

# b = PhaseApplication('/Users/isaigordeev/Desktop/2023/saxs/results/session_results/2023-07-28/22:57:00_prominence_kernel.json', AbstractPhaseKernel)
# b.phase_classification()

# a = Manager(peak_kernel=ParabolePeakKernel, phase_kernel=DefaultPhaseKernel)
a = DirectoryManager(peak_kernel=RobustParabolePeakKernel, phase_kernel=DefaultPhaseKernel)
# a = Manager(peak_data_directory='res/', peak_kernel=RobustParabolePeakKernel, phase_kernel=DefaultPhaseKernel)

# a = Manager(peak_kernel=ProminencePeakKernel, phase_kernel=DefaultPhaseKernel)
a()

# if __main__ == '__main__'
# a()
# test.directory_processing()