import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from saxs.gaussian_processing.functions import background_hyberbole
from saxs.gaussian_processing.peak.peak_kernel.parabole_kernel import ParabolePeakKernel


class RobustParabolePeakKernel(ParabolePeakKernel):
    str_type = "RobustParabolePeakKernel"

    def __init__(self, data_dir, file_analysis_dir,
                 is_preprocessing=True,
                 is_postprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 is_peak_processing=True
                 ):

        super().__init__(data_dir, file_analysis_dir,
                         is_preprocessing,
                         is_postprocessing,
                         is_background_reduction,
                         is_filtering,
                         is_peak_processing
                         )
        self.fitted_peak_params = None
        self.final_peaks = None
        self.y_final_fit = None

    @classmethod
    def define_signature(cls):
        cls.str_type = "RobustParabolePeakKernel"

    def gaussian_sum(self, x, *params):
        assert len(params) % 3 == 0
        y = np.zeros_like(x)
        number = 0
        for i in range(0, len(params), 3):
            mean, amplitude, std_dev = params[i:i + 3]
            # y += amplitude * np.exp(-((x - self.I_background_filtered[self.peaks[number]]) / std_dev) ** 2)
            y += amplitude * np.exp(-((x - mean) / std_dev) ** 2)
            number += 1
        return y

    def sum_gaussian_total_fit(self):
        if len(self.peaks) != 0 and len(self.peak_params) != 0:
            def loss_function(params):
                # y_pred = gaussian_sum(self.q, *params)
                y_pred = self.gaussian_sum(self.current_q_state, *params)

                # return np.sum((y_pred - self.I_background_filtered) ** 2)
                return np.sum((y_pred - self.I_background_filtered) ** 2)

            result = minimize(loss_function, self.peak_params, method='BFGS')
            fitted_params = result.x
            self.fitted_peak_params = fitted_params
            self.y_final_fit = self.gaussian_sum(self.current_q_state, *fitted_params)
            # print("fit", self.fitted_peak_params)

            self.robust_parabole_peak_kernel_plot()
            self.final_peaks = sorted(self.fitted_peak_params[::3])

    def postprocessing(self):
        self.sum_gaussian_total_fit()

    def peak_coordinates_fit_update(self):
        self.peaks = (self.final_peaks/self.delta_q).astype(int)

    def gathering(self) -> dict:
        peak_number = len(self.peaks) if self.peaks is not None else -1
        print("peak found: ", peak_number)

        return {
            'peak kernel method': self.class_info(),
            'peak_number': peak_number,
            'initial peak indices': self.peaks.tolist(),
            'q': self.final_peaks,
            # 'I': self.current_I_state[self.peaks].tolist(),
            # 'kernel': self.str_type
            # 'dI': dI.tolist(),
            # 'I_raw': I_raw.tolist(),
            # 'peaks': peaks_detected.tolist(),
            # 'params_amplitude': self.params.tolist()[::3],
            # 'params_mean': self.params.tolist()[1::3],
            # 'params_sigma': self.params.tolist()[2::3],
            # 'start_loss': self.start_loss,
            # 'final_loss': self.final_loss,
            # 'error': error
            # 'loss_ratio': self.final_loss / self.start_loss
        }

    def robust_parabole_peak_kernel_plot(self):
        plt.clf()
        plt.plot(self.current_q_state, self.total_fit, color='b', label="preliminary fit")
        plt.plot(self.current_q_state, self.I_background_filtered, color='g', label="unfiltered cut background")
        plt.plot(self.current_q_state, self.y_final_fit, color='r', label="final fit")

        plt.legend()

        plt.savefig("{}/robust_parabole_peak_kernel_plot.pdf".format(self.file_analysis_dir))


class RobustParabolePeakKernelWithBackground(RobustParabolePeakKernel):
    str_type = "RobustParabolePeakKernelWithBackground"

    def __init__(self, data_dir, file_analysis_dir,
                 is_preprocessing=True,
                 is_postprocessing=True,
                 is_background_reduction=True,
                 is_filtering=True,
                 is_peak_processing=True
                 ):
        super().__init__(data_dir, file_analysis_dir,
                         is_preprocessing,
                         is_postprocessing,
                         is_background_reduction,
                         is_filtering,
                         is_peak_processing
                         )
        self.final_background = None

    def default_background_reduction(self, background_function=background_hyberbole):
        super().default_background_reduction(background_function)
        self.peak_params = np.append(self.peak_params, self.popt_background[0])
        self.peak_params = np.append(self.peak_params, self.popt_background[1])

    def gaussian_sum_and_background(self, x, *params):
        return background_hyberbole(x, params[0], params[1]) + self.gaussian_sum(x, *(params[2:]))

    def sum_gaussian_and_background_total_fit(self):
        if len(self.peaks) != 0 and len(self.peak_params) != 0:
            def loss_function(params):
                y_pred = self.gaussian_sum_and_background(self.current_q_state, *params)

                return np.sum((y_pred - self.I_cut) ** 2)

            result = minimize(loss_function, self.peak_params, method='BFGS')
            fitted_params = result.x
            self.fitted_peak_params = fitted_params
            self.y_final_fit = background_hyberbole(self.current_q_state, self.fitted_peak_params[0],
                                                    self.fitted_peak_params[1]) + self.gaussian_sum(
                self.current_q_state, *(fitted_params[2:]))

            self.robust_parabole_peak_kernel_plot()
            self.robust_parabole_peak_kernel_plot_detailed()
            self.final_peaks = sorted(self.fitted_peak_params[2:][::3])

    def robust_parabole_peak_kernel_plot_detailed(self):
        plt.clf()
        print(self.popt_background)
        print(self.fitted_peak_params[:2])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        background = background_hyberbole(self.current_q_state, self.popt_background[0], self.popt_background[1])
        final_background = background_hyberbole(self.current_q_state, self.fitted_peak_params[0],
                                                self.fitted_peak_params[1])

        plt.plot(self.current_q_state, self.total_fit + background, color=colors[0], label="preliminary fit")
        plt.plot(self.current_q_state, self.I_cut, color=colors[1], label="unfiltered cut")
        plt.plot(self.current_q_state, background, color=colors[2], label="initial background")
        plt.plot(self.current_q_state, self.I_background_filtered, color=colors[3], label="background filtered")
        plt.plot(self.current_q_state, final_background, color=colors[4], label="final background")
        plt.plot(self.current_q_state, self.y_final_fit, color=colors[5], label="final fit")

        plt.legend()

        plt.savefig(f"{self.file_analysis_dir}/robust_parabole_peak_kernel_plot_with_background_detailed.pdf")

    def robust_parabole_peak_kernel_plot(self):
        plt.clf()
        print(self.popt_background)
        print(self.fitted_peak_params[:2])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        final_background = background_hyberbole(self.current_q_state, self.fitted_peak_params[0],
                                                self.fitted_peak_params[1])

        plt.plot(self.current_q_state, self.I_cut, color=colors[1], label="unfiltered cut")
        plt.plot(self.current_q_state, final_background, color=colors[4], label="final background")
        plt.plot(self.current_q_state, self.y_final_fit, color=colors[5], label="final fit")

        plt.legend()

        plt.savefig(f"{self.file_analysis_dir}/robust_parabole_peak_kernel_plot_with_background.pdf")

    def gaussian_fit_troubleshoot(self, accuracy=4):
        if len(self.peaks) != 0 and len(self.peak_params) != 0:
            grad = np.diff(self.peaks)
            peaks_to_delete = []
            peak_params_to_delete = []

            for i, x in enumerate(grad):
                if x < accuracy:
                    middle_i = self.peak_params[2 + 3 * i + 2]
                    middle_i_plus_1 = self.peak_params[
                        2 + 3 * (i + 1) + 2]

                    if middle_i > middle_i_plus_1:
                        self.peaks[i] = int(np.mean([self.peaks[i], self.peaks[i + 1]]))
                        peaks_to_delete.append(i)
                        peak_params_to_delete.extend(range(2 + 3 * i, 2 + 3 * i + 3))
                    else:
                        self.peaks[i] = int(np.mean([self.peaks[i], self.peaks[i + 1]]))
                        peaks_to_delete.append(i + 1)
                        peak_params_to_delete.extend(
                            range(2 + 3 * (i + 1), 2 + 3 * (i + 1) + 3))

            self.peaks = np.delete(self.peaks, peaks_to_delete)
            self.peak_params = np.delete(self.peak_params, peak_params_to_delete)

    def final_plot_from_fitted_params(self, number=0):
        plt.clf()
        final_background = background_hyberbole(self.q_raw, self.fitted_peak_params[0],
                                                self.fitted_peak_params[1])
        plt.plot(self.q_raw, self.I_raw-final_background, label='raw_plot_without_fitback')
        plt.plot(self.fitted_peak_params[2:][::3],
                 self.fitted_peak_params[2:][1::3], 'rx', label='peaks_on_raw')
        plt.plot(self.q_raw, self.zero_level, label='zero_level')
        plt.legend()
        plt.savefig("{}/final_plot_from_fitted_params_{}.pdf".format(self.file_analysis_dir, number))

    def postprocessing(self):
        self.gaussian_fit_troubleshoot()
        self.final_plot(1)
        self.sum_gaussian_and_background_total_fit()
        self.final_plot_from_fitted_params()
        # self.peak_coordinates_fit_update()

    def gathering(self) -> dict:
        peak_number = len(self.fitted_peak_params[2:])//3 if self.fitted_peak_params is not None and len(self.fitted_peak_params[2:]) % 3 == 0 else -1
        print("peak found: ", peak_number,len(self.fitted_peak_params))

        return {
            'peak kernel method': self.class_info(),
            'peak_number': peak_number,
            'initial peak indices': self.peaks.tolist(),
            'q': self.final_peaks,
            # 'I': self.current_I_state[self.peaks].tolist(),
            # 'kernel': self.str_type
            # 'dI': dI.tolist(),
            # 'I_raw': I_raw.tolist(),
            # 'peaks': peaks_detected.tolist(),
            'initial_params_amplitude': self.peak_params.tolist()[2:][::3],
            'initial_params_mean': self.peak_params.tolist()[2:][1::3],
            'initial_params_sigma': self.peak_params.tolist()[2:][2::3],

            'fitted_params_amplitude': self.fitted_peak_params.tolist()[2:][::3],
            'fitted_params_mean': self.fitted_peak_params.tolist()[2:][1::3],
            'fitted_params_sigma': self.fitted_peak_params.tolist()[2:][2::3],
            # 'start_loss': self.start_loss,
            # 'final_loss': self.final_loss,
            # 'error': error
            # 'loss_ratio': self.final_loss / self.start_loss
        }