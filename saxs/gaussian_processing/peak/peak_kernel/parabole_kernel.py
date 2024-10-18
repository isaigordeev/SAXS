import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize

from saxs.gaussian_processing.peak.peak_kernel.prominence_kernel import ProminencePeakKernel
from saxs.gaussian_processing.functions import parabole, gauss


class ParabolePeakKernel(ProminencePeakKernel):
    def __init__(self, data_dir, file_analysis_dir,
                 is_preprocessing=True,
                 is_postprocessing=False,
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

        self.parabole_number = 0
        self.gaussian_peak_number = 0
        self.parabole_popt_for_gauss = {}
        self.current_gauss = np.array([])
        self.peaks_processed = np.array([])
        self.peak_params = np.array([])  # mean, amlitude, std

    def negative_reduction(self):
        self.current_I_state = np.maximum(self.current_I_state, 0)

    def parabole_peak_minimize(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                # left_base = abs(self.peaks[i] - self.props['left_bases'][i])
                # right_base = abs(self.peaks[i] - self.props['right_bases'][i])

                # delta = min(left_base, right_base) / 2

                delta = 4
                start_delta = delta

                period1 = int(self.peaks[i] - delta)
                period2 = int(self.peaks[i] + delta)

                def loss_function(params):
                    sigma = params[0]
                    ampl = params[1]
                    y_pred = parabole(self.current_q_state, self.current_q_state[self.peaks[i]], sigma, ampl)
                    # return np.sum((y_pred - self.I_background_filtered) ** 2)
                    return np.sum((y_pred[period1:period2] - self.I_background_filtered[period1:period2]) ** 2)

                result = minimize(loss_function, np.array([0.5, 2]))
                fitted_params = result.x

                period2 = int(self.peaks[i] + start_delta)
                period1 = int(self.peaks[i] - start_delta)

                current_parabole = parabole(self.current_q_state, self.current_q_state[self.peaks[i]], fitted_params[0],
                                            fitted_params[1])

                plt.clf()
                plt.plot(self.current_q_state, self.current_I_state)
                # plt.plot(self.current_q_state[period1//2:period2*2], current_parabole[period1//2:period2*2])
                plt.plot(self.current_q_state[period1:period2], self.current_I_state[period1:period2])
                plt.plot(self.current_q_state[period1:period2], current_parabole[period1:period2], 'red')

                plt.title(f'{fitted_params}')
                plt.savefig('{}parabole_{}.png'.format(self.file_analysis_dir_peaks, self.parabole_number))

                self.parabole_number += 1
                self.parabole_popt_for_gauss[self.peaks[i]] = fitted_params

    def parabole_peak_curve_fit(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                # left_base = abs(self.peaks[i] - self.props['left_bases'][i])
                # right_base = abs(self.peaks[i] - self.props['right_bases'][i])

                # delta = min(left_base, right_base) / 2

                delta = 4
                start_delta = delta

                period1 = int(self.peaks[i] - delta)
                period2 = int(self.peaks[i] + delta)

                current_peak_parabole = lambda x, sigma, ampl: parabole(x, self.current_q_state[self.peaks[i]], sigma,
                                                                        ampl)

                popt, pcov = curve_fit(
                    f=current_peak_parabole,
                    xdata=self.current_q_state[period1:period2],
                    ydata=self.current_I_state[period1:period2],
                    bounds=([self.delta_q ** 2, 1], [0.05, 4 * self.max_I]),
                    sigma=self.dI[period1:period2]
                )

                period2 = int(self.peaks[i] + start_delta)
                period1 = int(self.peaks[i] - start_delta)

                current_parabole = current_peak_parabole(self.current_q_state, popt[0], popt[1])
                plt.clf()
                plt.plot(self.current_q_state, self.current_I_state)
                # plt.plot(self.current_q_state[period1//2:period2*2], current_parabole[period1//2:period2*2])
                plt.plot(self.current_q_state[period1:period2], self.current_I_state[period1:period2])
                plt.plot(self.current_q_state[period1:period2], current_parabole[period1:period2], 'red')

                plt.title(f'{popt},{np.sqrt(np.diag(pcov))}')
                plt.savefig('{}parabole_{}.png'.format(self.file_analysis_dir_peaks, self.parabole_number))

                self.parabole_number += 1
                self.parabole_popt_for_gauss[self.peaks[i]] = popt

    def gauss_peak_fitting_minimize(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                delta = self.parabole_popt_for_gauss[self.peaks[i]][0] / self.delta_q

                # delta /= 1.5

                period1 = int(self.peaks[i] - delta)
                period2 = int(self.peaks[i] + delta)
                padding = 0
                zeros_padding = np.zeros(padding)

                def loss_function(params):
                    background = np.array([])

                    sigma = params[0]
                    ampl = params[1]

                    y_pred = gauss(self.current_q_state, self.current_q_state[self.peaks[i]], sigma, ampl)
                    background = np.append(background, zeros_padding)
                    background = np.append(background, self.I_background_filtered[period1:period2])
                    background = np.append(background, zeros_padding)

                    # return np.sum((y_pred - self.I_background_filtered) ** 2)
                    return np.sum((y_pred[period1 - padding:period2 + padding] - background) ** 2)

                # result = minimize(loss_function, self.parabole_popt_for_gauss[self.peaks[i]], method='BFGS')
                result = minimize(loss_function, self.parabole_popt_for_gauss[self.peaks[i]])
                fitted_params = result.x

                self.peak_params = np.append(self.peak_params, self.current_q_state[self.peaks[i]])
                self.peak_params = np.append(self.peak_params, fitted_params[1])
                self.peak_params = np.append(self.peak_params, fitted_params[0])

                self.current_gauss = gauss(self.current_q_state, self.current_q_state[self.peaks[i]], fitted_params[0],
                                           fitted_params[1])

                plt.clf()
                plt.plot(self.current_q_state, self.current_I_state)
                plt.plot(self.current_q_state, self.current_gauss)
                # plt.plot(self.current_q_state[period1//2:period2*2], current_parabole[period1//2:period2*2])
                plt.plot(self.current_q_state[period1:period2], self.current_I_state[period1:period2])
                plt.plot(self.current_q_state[period1:period2], self.current_gauss[period1:period2], 'red')

                plt.title(f'{fitted_params}')
                plt.savefig(
                    '{}minimize_gauss_peak_{}.png'.format(self.file_analysis_dir_peaks, self.gaussian_peak_number))

                self.gaussian_peak_number += 1

    def gauss_peak_fitting_curve_fit(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:

                delta = self.parabole_popt_for_gauss[self.peaks[i]][0] / self.delta_q

                # delta /= 2

                period1 = int(self.peaks[i] - delta)
                period2 = int(self.peaks[i] + delta)

                if period1 < 0:
                    period1 = 0
                if period2 >= len(self.peaks):
                    period1 = len(self.peaks) - 1

                current_peak_gauss = lambda x, sigma, ampl: gauss(x, self.current_q_state[self.peaks[i]], sigma, ampl)
                # print("y_data", self.current_I_state)

                popt, pcov = curve_fit(
                    f=current_peak_gauss,
                    xdata=self.current_q_state[period1:period2],
                    ydata=self.current_I_state[period1:period2],
                    bounds=([self.delta_q ** 2, 1], [0.05, 4 * self.max_I]),
                    sigma=self.dI[period1:period2]
                )

                self.current_gauss = current_peak_gauss(self.current_q_state, popt[0], popt[1])

                self.peak_params = np.append(self.peak_params, self.current_q_state[self.peaks[i]])
                self.peak_params = np.append(self.peak_params, popt[1])
                self.peak_params = np.append(self.peak_params, popt[0])

                plt.clf()
                plt.plot(self.current_q_state, self.current_I_state)
                plt.plot(self.current_q_state, self.current_gauss)
                # plt.plot(self.current_q_state[period1//2:period2*2], current_parabole[period1//2:period2*2])
                plt.plot(self.current_q_state[period1:period2], self.current_I_state[period1:period2])
                plt.plot(self.current_q_state[period1:period2], self.current_gauss[period1:period2], 'red')

                plt.title(f'{popt},{np.sqrt(np.diag(pcov))}')
                plt.savefig('{}gauss_peak_{}.png'.format(self.file_analysis_dir_peaks, self.gaussian_peak_number))

                self.gaussian_peak_number += 1

    def gaussian_peak_reduction(self, i):
        self.current_I_state -= self.current_gauss
        self.total_fit += self.current_gauss
        self.peaks_processed = np.append(self.peaks_processed, self.peaks[i])

    def search_peaks(self, height=1, prominence=0.3):  # TODO Optimal parameteres?
        self.sequential_search_peaks(height, prominence)

    def sequential_search_peaks(self, height, prominence):

        peak_counter = 0

        while True:
            self.negative_reduction()

            super().search_peaks(height, prominence)
            if len(self.peaks) == 0:
                break

            self.parabole_peak_curve_fit(peak_counter)
            # self.parabole_peak_minimize(peak_counter)
            self.gauss_peak_fitting_minimize(peak_counter)
            # self.gauss_peak_fitting_curve_fit(peak_counter)
            self.gaussian_peak_reduction(peak_counter)

        self.peaks = self.peaks_processed.astype(int)


    def relevant_search_peaks(self):
        pass


