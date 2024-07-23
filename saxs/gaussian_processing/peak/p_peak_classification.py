import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks

from .peak_classificator import DefaultPeakApplication
from saxs.gaussian_processing.functions import gaussian_sum, moving_average, parabole
from saxs.gaussian_processing.settings_processing import INFINITY, PROMINENCE, BACKGROUND_COEF


class PDefaultPeakClassificator(DefaultPeakApplication):
    def __init__(self, current_session, data_path, filename):

        super().__init__(current_session, data_path, filename)

        self.ppeak_number = 0
        self.sigmas = np.array([])
        self.deltas = np.array([])
        self.data = {}
        self.mask_factor = 0

    def peak_searching(self, height=0, distance=5, prominence=0.1):
        self.peaks, self.peaks_data = find_peaks(self.difference,
                                                 height=height, #for sample
                                                 distance=distance,
                                                 # threshold=0.2,
                                                 plateau_size=1, #for sample
                                                 prominence=prominence
                                                  ) # NOTE attention
        # self.peaks, self.peaks_data = find_peaks(self.difference)


    # probably it makes sense just move the centres?

    def peak_verifying(self, i) -> bool:
        if len(self.peaks) > i:
            if self.peaks[i] in self.peak_previous:
                #     self.peaks = np.delete(self.peaks, *np.where(self.peaks == self.peak_previous))
                # self.peak_widths = peak_widths(self.difference, self.peaks, rel_height=0.6)
                return True

    def custom_peak_fitting_with_parabole(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                # left_base = abs(self.peaks[i] - self.peaks_data['left_bases'][i])
                # right_base = abs(self.peaks[i] - self.peaks_data['right_bases'][i])
                # delta = min(left_base, right_base) / 2
                # start_delta = delta

                delta = 3
                # print(left_base, right_base, 'bases')
                # period1 = self.peaks[i] - int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
                # period2 = self.peaks[i] + int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])

                y = self.I_cut_background_reduced
                # window_size = 5
                # smoothed_y = moving_average(y, window_size)

                # uniform_indices = np.linspace(0, len(self.I_background_filtered) - 1, 500)
                # uniform_array = np.interp(uniform_indices, range(len(self.I_background_filtered)), smoothed_y)
                # plt.clf()
                # plt.plot(uniform_indices, uniform_array, 'x')
                # plt.plot(self.q, self.I_background_filtered)

                # plt.show()
                # print(uniform_array)

                # sigma_values = np.linspace(3, 20, 5) # NOTE optimine

                sigma_values = [3]  # 3?
                best_metric = np.inf

                period1_global = int(self.peaks[i] - 20)
                period2_global = int(self.peaks[i] + 20)

                period1_fix = int(self.peaks[i] - 20)
                period2_fix = int(self.peaks[i] + 20)

                current_peak_parabole = lambda x, sigma, ampl: parabole(x, self.q[self.peaks[i]], sigma, ampl)

                popt = None

                p_num = 0
                for delta in sigma_values:

                    period1 = int(self.peaks[i] - delta)
                    period2 = int(self.peaks[i] + delta)
                    # print(period1, period2, delta, 'perods')

                    popt1, pcov1 = curve_fit(
                        f=current_peak_parabole,
                        xdata=self.q[period1:period2],
                        ydata=self.I_cut_background_reduced[period1:period2],
                        bounds=([self.delta_q ** 2, 1], [0.05, 4 * self.max_I]),
                        # sigma=self.dI[period1:period2]
                    )

                    # period1_fix = int(self.peaks[i] - popt1[0]/self.delta_q)
                    # period2_fix = int(self.peaks[i] + popt1[0]/self.delta_q)

                    period1_fix = int(self.peaks[i] - delta)
                    period2_fix = int(self.peaks[i] + delta)

                    new_delta = popt1[0] / self.delta_q
                    # print(new_delta, 'new delta')

                    smoothed_difference = current_peak_parabole(self.q, popt1[0], popt1[1])

                    # fixed_metric
                    # metric = np.mean(np.square(
                    #     smoothed_difference[period1_fix:period2_fix] - smoothed_y[period1_fix:period2_fix]))/(2*new_delta)

                    metric = np.mean(np.square(
                        smoothed_difference[period1_fix:period2_fix] - self.difference_start[
                                                                       period1_fix:period2_fix])) / (delta)

                    # plt.clf()
                    # plt.plot(self.q, self.I_background_filtered)
                    # plt.plot(self.q[period1_global:period2_global], self.difference_start[period1_global:period2_global])
                    # plt.plot(self.q[period1_global:period2_global], smoothed_difference[period1_global:period2_global])
                    # plt.savefig(('heap/parabole_cur_' + str(p_num) + '.png'))

                    # p_num += 1

                    # print(metric)
                    if metric < best_metric:
                        best_metric = metric
                        start_delta = delta
                        popt = popt1
                        pcov = pcov1

                period2 = int(self.peaks[i] + start_delta)
                period1 = int(self.peaks[i] - start_delta)

                best_delta = popt[0] / self.delta_q

                period2 = int(self.peaks[i] + best_delta / 2)
                period1 = int(self.peaks[i] - best_delta / 2)

                # print(start_delta, 'best delta')

                current_parabole = current_peak_parabole(self.q, popt[0], popt[1])[period1:period2]
                plt.clf()
                plt.plot(self.q, self.I_cut_background_reduced)
                plt.plot(self.q[period1:period2], self.I_cut_background_reduced[period1:period2], '.')
                # plt.plot(self.q, smoothed_y, label='smooth gen')
                plt.plot(self.q[period1:period2], current_parabole, label='smooth')
                # plt.plot(self.q[period1:period2], current_parabole, 'x')
                plt.legend()
                plt.title(f'{popt},{np.sqrt(np.diag(pcov))}')
                # print({popt[0]/self.delta_q})
                # plt.savefig(('heap/parabole_' + str(p_num) + '.png'))
                plt.savefig(('heap/parabole_' + str(self.ppeak_number) + '.pdf'))


                self.ppeak_number += 1
                self.deltas = np.append(self.deltas, popt[0] / self.delta_q)
                self.sigmas = np.append(self.sigmas, popt[0])
                self.data[self.peaks[i]] = popt

                # return gauss(self.q, popt[0], popt[1]), \
                #     period1, period2, i, \
                #     self.q[self.peaks[i]], \
                #     gauss(self.q, popt[0], popt[1])[self.peaks[i]], popt[0], popt[1], self.peaks[i], perr

    def custom_peak_fitting(self, i, width_factor=1):
        if len(self.peaks) > i:
            delta = abs(self.data[self.peaks[i]][0] / self.delta_q) / 2
            # delta = min(abs(self.peaks[i] - self.peaks_data['left_bases'][i]),
            #             abs(self.peaks[i] - self.peaks_data['right_bases'][i]))
            # period1 = self.peaks[i] - int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
            # period2 = self.peaks[i] + int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
            period1 = int(self.peaks[i] - delta)
            period2 = int(self.peaks[i] + delta)

            y = self.I_cut_background_reduced
            window_size = 5
            smoothed_y = moving_average(y, window_size)

            gauss = lambda x, c, b: c * np.exp(-(x - self.q[self.peaks[i]]) ** 2 / (b ** 2))
            print(self.peaks[i])
            print(delta)
            print(self.difference[period1:period2])
            if period1 != period2:
                popt, pcov = curve_fit(
                    f=gauss,
                    xdata=self.q[period1:period2],
                    # ydata=smoothed_y[period1:period2], # NOTE strangely works
                    ydata=self.difference[period1:period2],
                    p0=(self.data[self.peaks[i]][1], self.data[self.peaks[i]][0]), # TODO initial conditions and better fitting corresponding to the parabole
                    bounds=(self.delta_q ** 4, [2 * 2 * self.max_I, 1, ]),
                    # sigma=self.dI[period1:period2]
                )

                perr = np.sqrt(np.diag(pcov))

                self.params = np.append(self.params, popt[0])
                self.params = np.append(self.params, self.q[self.peaks[i]])
                self.params = np.append(self.params, popt[1])

                # plt.clf()
                # plt.plot(self.q, gauss(self.q, popt[0], popt[1]))
                # plt.plot(self.q, self.I_background_filtered)
                # plt.plot(self.q[period1:period2], self.I_background_filtered[period1:period2])
                # plt.savefig('heap/' + str(self.peak_number) + '.png')

                return gauss(self.q, popt[0], popt[1]), \
                    period1, period2, i, \
                    self.q[self.peaks[i]], \
                    gauss(self.q, popt[0], popt[1])[self.peaks[i]], popt[0], popt[1], self.peaks[i], perr


    def peak_substraction(self, i):
        self.peak_empty = False

        factor = 1
        self.custom_peak_fitting_with_parabole(i)
        peak = self.custom_peak_fitting(i)

        if peak is None:
            self.peak_empty = True
            print('peak error empty')
            return 0

        # while peak[9][1] / peak[7] > 0.2:
        #     factor *= 0.9
        #     peak = self.peak_fitting_gauss(i, width_factor=factor)
        #     if peak is None:
        #         self.peak_empty = True
        #         return

        self.peak_plots[self.peak_number] = peak[0]

        # more efficient O(1) – previsioned array
        valid_zone = np.arange(-5, 6, 1)  # TODO
        for x in valid_zone:
            self.peak_previous = np.append(self.peak_previous, self.peaks[i] + x)

        self.peaks_analysed = np.append(self.peaks_analysed,
                                        (peak[4],
                                         peak[5]))
        self.peaks_analysed_q = np.append(self.peaks_analysed_q,
                                          peak[4])
        self.peaks_analysed_I = np.append(self.peaks_analysed_I,
                                          peak[5])
        self.peaks_analysed_b = np.append(self.peaks_analysed_b,
                                          peak[7])
        # self.widths = np.append(self.peak_widths,
        #                         self.peak_widths[0][0])

        self.peaks_analysed_dict[peak[4]] = peak[5]
        self.peaks_detected = np.append(self.peaks_detected, peak[8])

        self.total_fit += peak[0]
        self.difference -= peak[0]
        self.filtering_negative()
        self.stage_plot()
        self.peak_number += 1
        # self.peaks_boundaries = np.append(self.peaks_boundaries, (peak[0], peak[1]))


    def stage_plot(self):
        plt.clf()
        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data')
        plt.plot(self.q, self.difference_start, label='filtered_raw_data')
        plt.plot(self.q, self.difference, 'x', label='filtered_data')
        plt.plot(self.q[self.peaks], self.difference_start[self.peaks], "x", label='all_peaks_detected')
        # if self.peak_fitting_gauss(i) is not None:
        #     plt.plot(self.q, self.peak_fitting_gauss(i)[0], linewidth=2.5, label='current_peak')
        #     plt.plot(self.q[self.peak_fitting_gauss(i)[1]:self.peak_fitting_gauss(i)[2]],
        #              self.difference[self.peak_fitting_gauss(i)[1]:self.peak_fitting_gauss(i)[2]], 'o',
        #              label='zone_curr_peak')

        plt.plot(self.q, self.peak_plots[self.peak_number], 'x', linewidth=2.5, label='current_peak')
        plt.plot(self.q, self.zeros, label='zero_level')
        plt.plot(self.q, self.total_fit, linewidth=2.5, label='total')
        plt.legend()
        plt.savefig(self.file_analyse_dir_peaks + '/' + self.filename + '_peak:' + str(self.peak_number) + '.pdf')

    def result_plot(self):
        plt.clf()
        self.peaks_detected = self.peaks_detected.astype(int)

        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data_without_back')
        plt.plot(self.q, self.I_cut_background_reduced, label='filtered_raw_data_without_back')
        # plt.plot(self.q, self.difference, label='filtered_data')
        plt.plot(self.q, self.zeros, label='zero_level')
        # plt.plot(self.q[self.peaks_detected], self.I_background_filtered[self.peaks_detected], 'x',
        #          label='peaks_on_raw_without_back')
        # plt.plot(self.q[self.peaks_detected], self.difference_start[self.peaks_detected], 'x',
        #          label='peaks_on_filtered_without_back')
        for x in range(self.peak_number):
            plt.plot(self.q, self.peak_plots[x])
        plt.plot(self.q, self.total_fit, linewidth=2, label='total')
        plt.legend()
        plt.savefig(self.file_analyse_dir + '/10_result_' + self.filename + '.pdf')

        plt.clf()
        plt.plot(self.q, self.I, label='raw_data')
        plt.plot(self.q[self.peaks_detected], self.I[self.peaks_detected], 'x', label='peaks_on_raw')
        plt.plot(self.q, np.zeros(len(self.q)), label='zero_level')
        plt.legend()
        plt.savefig(self.file_analyse_dir + '/11_result_raw_' + self.filename + '.pdf')

        # plt.clf()
        # plt.plot(self.q, self.I, label='raw_data')
        # plt.plot(self.q[self.peaks_detected], self.I[self.peaks_detected], 'x', label='peaks_on_raw')
        # plt.plot(self.q, np.zeros(len(self.q)), label='zero_level')
        # plt.legend()
        # plt.savefig(self.file_analyse_dir + '/11_result_raw_' + self.file + '.pdf')

    def peak_processing(self, number_peak=INFINITY, get=False):

        while number_peak > 0:
            current_peak = 0
            self.peak_searching(height=0, prominence=PROMINENCE, distance=10)  # TODO good parameters and metric of suspicious peaks
            if len(self.peaks) != 0:
                while len(self.peaks) > current_peak and number_peak > 0:
                    # self.custom_peak_searching()
                    # self.peak_searching(height=0, prominence=PROMINENCE, distance=6)
                    # if self.peak_verifying(current_peak):
                    number_peak -= 1
                    self.peak_substraction(current_peak)
                    current_peak += 1
            else:
                break
            # print('peak_number', self.peak_number)

    def gaussian_sum_non_fit_q(self, x, *params):
        y = np.zeros_like(x)
        number = 0
        for i in range(0, len(params), 3):
            mean, amplitude, std_dev = params[i:i + 3]
            y += amplitude * np.exp(-((x - self.peaks_analysed_q[number]) / std_dev) ** 2)
            number += 1
        return y

    def sum_total_fit(self):
        if (len(self.params) != 0):
            print(self.params)

            def loss_function(params):
                # y_pred = gaussian_sum(self.q, *params)
                y_pred = self.gaussian_sum_non_fit_q(self.q, *params)

                # return np.sum((y_pred - self.I_background_filtered) ** 2)
                return np.sum((y_pred - self.smoothed_I) ** 2)

            result = minimize(loss_function, self.params, method='BFGS')
            fitted_params = result.x
            self.params = fitted_params
            y_fit = gaussian_sum(self.q, *fitted_params)

            plt.clf()
            plt.title(str(sorted(self.params.tolist()[1::3])))
            plt.plot(self.q, self.I_cut_background_reduced, 'g--', label='raw')
            plt.plot(self.q, y_fit, 'r-', label='found ' + str(self.peak_number))

            for x in self.peaks_x:
                plt.axvline(x, color='red', linestyle='--', label='Vertical Line')

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')

            plt.savefig(self.file_analyse_dir + '/xx_total_fit_' + self.filename + '.pdf')
            # plt.show()

        else:
            plt.plot(self.q, self.I_cut_background_reduced, 'g--', label='not found')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(self.file_analyse_dir + '/xx_not_found_' + self.filename + '.pdf')

    def postprocessing(self):
        to_delete = []

        self.peaks_analysed_q = sorted(self.peaks_analysed_q)
        for i in range(len(self.peaks_analysed_q)-1):
            # print(abs(self.peaks_analysed_q[i] - self.peaks_analysed_q[i + 1]))
            if abs(self.peaks_analysed_q[i] - self.peaks_analysed_q[i+1]) < 0.002:
                to_delete.append(i+1)

        # print(to_delete)
        self.peaks_analysed_q = np.delete(self.peaks_analysed_q, to_delete)
        self.peaks_analysed_I = np.delete(self.peaks_analysed_I, to_delete)

    def gathering(self):
        # print('Covariance raw', np.cov(self.difference_start, self.total_fit)[0][1])
        # print('Covariance filtered', np.cov(self.I_backfiltered, self.total_fit)[0][1])
        self.final_loss = np.mean((self.difference_start - self.total_fit) ** 2)
        # print('Covariance raw', np.mean((self.I_backfiltered-self.total_fit)**2))
        # print('Loss raw', self.start_loss)
        # print('Loss filtered', self.final_loss)

        sorted_indices_q = np.argsort(self.peaks_analysed_q)

        I = self.peaks_analysed_I[sorted_indices_q]
        q = self.peaks_analysed_q[sorted_indices_q]
        # I_raw = self.I[self.peaks_detected][sorted_indices_q]
        # dI = self.dI[self.peaks_detected][sorted_indices_q]
        # peaks_detected = self.peaks_detected[sorted_indices_q]
        error = np.sum(self.total_fit - self.I_cut_background_reduced) / np.sum(self.I_cut_background_reduced)
        # print(error, 'Error')
        print('WRITING {}'.format(self.filename))
        return {
            'peak_number': self.peak_number,
            'q': q.tolist(),
            # 'I': I.tolist(),
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
