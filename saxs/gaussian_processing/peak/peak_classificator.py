import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, medfilt, savgol_filter

from saxs.gaussian_processing.functions import background_hyberbole, gaussian_sum, moving_average, parabole
from saxs.gaussian_processing.peak.peak_application import PeakApplication
from saxs.gaussian_processing.settings_processing import INFINITY, PROMINENCE, BACKGROUND_COEF, START, \
    WINDOWSIZE, \
    RESOLUTION_FACTOR




class DefaultPeakApplication(PeakApplication):
    def __init__(self, current_session, data_path, filename):
        super().__init__(current_session, data_path, filename)

        # self.peaks_plots = {}
        self.cut_point = None
        self.pcov_background = None
        self.popt_background = []
        self.zeros = np.zeros(len(self.q))
        self.q_cut = None
        self.I_cut = None

        ##########
        self.I_filt = np.array([])
        self.resolution = 0.5
        self.best_sigma = None
        self.params = np.array([])
        self.widths = np.array([])
        self.peak_plots = {}
        self.peaks_plots = np.array([])
        self.I_cut_background_reduced = np.array([])
        self.popt = []
        self.pcov = []
        self.background = []
        self.difference = []
        self.peaks = []
        self.difference_start = []
        self.peaks_data = None
        self.peak_widths = np.array([])
        self.peak_previous = np.array([])
        self.zeros = np.zeros(len(self.q))
        self.peaks_analysed = np.array([])
        self.peaks_analysed_q = np.array([])
        self.peaks_analysed_I = np.array([])
        self.peaks_analysed_b = np.array([])
        self.outnumbered_peaks = np.array([])
        self.peak_number = 0
        self.peaks_analysed_dict = {}
        self.total_fit = []
        self.peaks_detected = np.array([])
        self.start_loss = 0
        self.final_loss = 0
        self.passed = 0
        self.gauss = True
        ######################


        self.cutting_noisy_by_default()
        self.simple_background()

    def setting_state(self):
        self.difference = self.I_cut_background_reduced
        self.difference_start = self.I_cut_background_reduced

    def cutting_noisy_by_default(self):
        self.cut_point = np.argmax(self.raw > START)
        self.q_cut, self.I_cut = self.q[self.cut_point:], self.I_raw[self.cut_point:],
        # self.dI = self.dI[i:]

        self.max_I = np.max(self.I_cut)


    def simple_background(self):

        popt, pcov = curve_fit(
            f=background_hyberbole,
            xdata=self.q_cut,
            # ydata=self.I,
            ydata=self.I_cut,
            p0=(3, 2),
            sigma=self.dI
        )

        self.popt_background = popt
        self.pcov_background = pcov

        self.background = background_hyberbole(self.q_cut, self.popt_background[0], self.popt_background[1])
        # self.I_background_filtered = self.I - BACKGROUND_COEF * self.saxs_model
        self.I_cut_background_reduced = self.I_cut - BACKGROUND_COEF * self.background



    def filtering(self):
        if self.gauss:
            self.I_background_reduced = np.concatenate((np.zeros(self.cut_point), self.I_cut_background_reduced))

            # y = self.I_cut_background_reduced
            y = self.I_background_reduced
            window_size = 10

            self.difference = moving_average(y, window_size)
            self.difference_start = moving_average(y, window_size)

            # self.difference = gaussian_filter(self.I_background_filtered,
            #                                   sigma=SIGMA_FILTER,
            #                                   truncate=TRUNCATE,
            #                                   cval=0)
            # self.difference_start = gaussian_filter(self.I_background_filtered,
            #                                         sigma=SIGMA_FILTER,
            #                                         truncate=TRUNCATE,
            #                                         cval=0)

            # plt.plot(self.I_cut_background_reduced)
            # plt.plot(self.I_cut)
            peaks, _ = find_peaks(self.difference, height=1, prominence=0.5)

            plt.plot(self.q, self.difference)
            plt.plot(self.q[peaks], self.difference[peaks], 'x')
            plt.show()

            self.I_background_reduced = np.concatenate((np.zeros(self.cut_point), self.I_cut_background_reduced))
            peaks, _ = find_peaks(self.I_background_reduced, height=1, prominence=1)
            print(_["left_bases"])
            print(_["right_bases"])
            print(peaks)
            plt.plot(self.q, self.I_background_reduced)
            plt.plot(self.q[_["right_bases"][0]], self.I_background_reduced[_["left_bases"][0]], 'ro')

            plt.plot(self.q[peaks], self.I_background_reduced[peaks], 'x')
            plt.show()

            # self.start_loss = np.mean((self.difference_start - self.total_fit) ** 2)
    def noisy_parts_detection(self):
        peaks, properties = find_peaks(self.I_background_reduced, height=1, prominence=1)
        return properties['right_bases'][0]

    def denoising(self):
        noisy_indice = self.noisy_parts_detection()

        # noisy_part = np.ones(noisy_indice)
        noisy_part = moving_average(self.I_background_reduced[:noisy_indice], 10)

        # noiseless_part = medfilt(self.I_background_reduced[noisy_indice:], 3)
        noiseless_part = self.I_background_reduced[noisy_indice:]

        self.I_denoised = medfilt(np.concatenate((noisy_part, noiseless_part)), 3)
        # self.I_filt = medfilt(good_smoothed_without_loss, 3)


        # _, __ = find_peaks(self.I_denoised, height=1, prominence=1)
        # plt.plot(self.q, self.I_denoised)
        # plt.plot(self.q, self.I_background_reduced)
        # plt.plot(self.q[_], self.I_background_reduced[_], 'ro')
        # plt.show()
        # plotting
        # plt.plot(self.q[noisy_indice:], medfilt(self.I_background_reduced[noisy_indice:], 3))
        # plt.plot(self.q[:noisy_indice], noisy_part)
        # plt.plot(self.q[noisy_indice:], noiseless_part)
        # plt.show()

    def prefiltering_(self):

        noisy_indices = self.noisy_parts_detection()
        print(noisy_indices)
        first_part = medfilt(self.I[:max(noisy_indices)], 3)
        first_part = savgol_filter(self.I[:max(noisy_indices)], 15, 4, deriv=0)

        # first_part = gaussian_filter(first_part, sigma=10)

        # first_part = gaussian_filter(self.I[:max(noisy_indices)], sigma=10)
        # first_part = medfilt(first_part, 3)
        print('FILTERING {}'.format(self.filename))

        sec_part = medfilt(self.I[max(noisy_indices):], 3)
        good_smoothed_without_loss = np.concatenate((first_part, sec_part))

        # return medfilt(good_smoothed_without_loss, 3)

        self.I_filt = medfilt(good_smoothed_without_loss, 3)

    def background_reduction(self):
        # self.I_filt = self.prefiltering()

        # self.I_filt = self.I_filt[i:]

        self.zeros = np.zeros(len(self.q))
        self.total_fit = self.zeros
        self.peaks_plots = np.zeros((20, len(self.q)))

        popt, pcov = curve_fit(
            f=background_hyberbole,
            xdata=self.q,
            # ydata=self.I,
            ydata=self.I_filt,
            p0=(3, 2),
            sigma=self.dI
        )

        self.popt_background = popt
        self.background = background_hyberbole(self.q, self.popt_background[0], self.popt_background[1])
        # self.I_background_filtered = self.I - BACKGROUND_COEF * self.saxs_model
        self.I_cut_background_reduced = self.I_filt - BACKGROUND_COEF * self.background

        # self.difference = savgol_filter(I - background_coef * self.saxs_model, 15, 4, deriv=0)
        # self.start_difference = savgol_filter(I - background_coef * self.saxs_model, 15, 4, deriv=0)


    def custom_filtering(self):
        y = self.I_cut_background_reduced
        self.smoothed_I = moving_average(y, WINDOWSIZE)

        sigma_values = np.linspace(0.5, 5.0, 10)

        self.best_sigma = None
        best_metric = np.inf

        for sigma in sigma_values:
            smoothed_difference = gaussian_filter(self.I_cut_background_reduced,
                                                  sigma=sigma,
                                                  truncate=4.0)

            metric = np.mean(np.square(smoothed_difference - self.smoothed_I))

            if metric < best_metric:
                best_metric = metric
                best_sigma = sigma

        # print("Best SIGMA: ", best_sigma)
        # print("Best SIGMA_Metric: ", best_metric)

        plt.clf()
        self.difference_start = gaussian_filter(self.I_cut_background_reduced,
                                                sigma=best_sigma,
                                                truncate=4.0)
        self.difference = gaussian_filter(self.I_cut_background_reduced,
                                          sigma=best_sigma,
                                          truncate=4.0)

    def custom_filtering_(self):
        smoothed_data = medfilt(self.I_background_reduced, 3)

        plt.plot(self.I_background_reduced)
        plt.plot(smoothed_data)
        plt.show()
        plt.clf()

    def background_plot(self):
        plt.clf()
        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data_without_background')
        plt.plot(self.q, self.background, label='background')
        plt.plot(self.q, BACKGROUND_COEF * self.background, label='moderated_background')
        plt.plot(self.q, self.I, linewidth=0.5, c='b', label='raw_data')
        plt.plot(self.q, self.zeros, label='zero_level')
        plt.legend()
        plt.savefig(self.file_analyse_dir + '/00_background_raw_' + self.filename + '.pdf')

        plt.clf()
        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data')
        plt.plot(self.q, self.difference_start, label='filtered_raw_data')
        plt.plot(self.q, self.zeros, label='zero_level')
        plt.legend()
        plt.savefig(self.file_analyse_dir + '/01_background_filtered_' + self.filename + '.pdf')

    def peak_searching(self, height=0, distance=5, prominence=0.1):
        self.peaks, self.peaks_data = find_peaks(self.difference,
                                                 height=height,
                                                 distance=distance,
                                                 threshold=0.2,
                                                 plateau_size=1,
                                                 prominence=(prominence, None))
        # if self.peaks.size == 0:
        # print(self.peaks_data)

    def peak_verifying(self, i):
        if len(self.peaks) > i:
            if self.peaks[i] in self.peak_previous:
                #     self.peaks = np.delete(self.peaks, *np.where(self.peaks == self.peak_previous))
                # self.peak_widths = peak_widths(self.difference, self.peaks, rel_height=0.6)
                return True

    def custom_peak_fitting_with_parabole(self, i):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                left_base = abs(self.peaks[i] - self.peaks_data['left_bases'][i])
                right_base = abs(self.peaks[i] - self.peaks_data['right_bases'][i])
                delta = min(left_base, right_base) / 2
                start_delta = delta

                delta = 3
                print(left_base, right_base, 'bases')
                # period1 = self.peaks[i] - int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
                # period2 = self.peaks[i] + int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])

                y = self.I_cut_background_reduced
                window_size = 5
                smoothed_y = moving_average(y, window_size)

                sigma_values = np.linspace(3, 20, 5)  # NOTE optimine

                best_metric = np.inf

                period1 = int(self.peaks[i] - delta)
                period2 = int(self.peaks[i] + delta)

                print(period1, period2, 'perods')

                current_peak_parabole = lambda x, sigma, ampl: parabole(x, self.q[self.peaks[i]], sigma, ampl)

                popt, pcov = curve_fit(
                    f=current_peak_parabole,
                    xdata=self.q[period1:period2],
                    ydata=self.difference_start[period1:period2],
                    bounds=([self.delta_q ** 2, 1], [0.05, 4 * self.max_I]),
                    sigma=self.dI[period1:period2]
                )

                print(popt)
                print(pcov)

                period2 = int(self.peaks[i] + start_delta)
                period1 = int(self.peaks[i] - start_delta)

                current_parabole = current_peak_parabole(self.q, popt[0], popt[1])[period1:period2]
                plt.clf()
                plt.plot(self.q[period1:period2], current_parabole)
                plt.plot(self.q[period1:period2], self.I_cut_background_reduced[period1:period2], 'x')
                plt.plot(self.q, self.I_cut_background_reduced)
                plt.title(f'{popt},{np.sqrt(np.diag(pcov))}')
                print({popt[0] / self.delta_q})
                plt.savefig(('heap/parabole_' + str(self.peak_number) + '.png'))

                # return gauss(self.q, popt[0], popt[1]), \
                #     period1, period2, i, \
                #     self.q[self.peaks[i]], \
                #     gauss(self.q, popt[0], popt[1])[self.peaks[i]], popt[0], popt[1], self.peaks[i], perr

    def custom_peak_fitting(self, i, width_factor=1):
        if len(self.peaks) > i:
            if np.size(self.peaks) != 0:
                left_base = abs(self.peaks[i] - self.peaks_data['left_bases'][i])
                right_base = abs(self.peaks[i] - self.peaks_data['right_bases'][i])
                delta = min(left_base, right_base)
                # period1 = self.peaks[i] - int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
                # period2 = self.peaks[i] + int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
                period1 = self.peaks[i] - delta
                period2 = self.peaks[i] + delta

                start_delta = delta

                gauss = lambda x, ampl, sigma: ampl * np.exp(-(x - self.q[self.peaks[i]]) ** 2 / (sigma ** 2))

                y = self.I_cut_background_reduced
                window_size = 5
                smoothed_y = moving_average(y, window_size)

                self.resolution = 0.5

                sigma_values = np.linspace(4, start_delta, 10)

                best_metric = np.inf

                for delta in sigma_values:
                    statement = max(right_base, left_base) > 40 and max(right_base, left_base) / min(right_base,
                                                                                                     left_base) > 3
                    if statement:
                        self.resolution = 1

                    period1 = int(self.peaks[i] - delta)
                    period2 = int(self.peaks[i] + delta)

                    if period1 != period2:

                        popt, pcov = curve_fit(
                            f=gauss,
                            xdata=self.q[period1:period2],
                            ydata=self.difference[period1:period2],
                            bounds=(self.delta_q ** 4, [2 * 2 * self.max_I, 1, ]),
                            sigma=self.dI[period1:period2]
                        )

                        smoothed_difference = gauss(self.q, popt[0], popt[1])

                        # metric = self.resolution*np.mean(np.square(smoothed_difference[period1:period2] - self.difference[period1:period2]))-np.sqrt(delta)/self.resolution
                        metric = self.resolution * np.mean(np.square(
                            smoothed_difference[period1:period2] - self.difference[period1:period2])) - (
                                             1 - self.resolution) * np.sqrt(
                            delta ** 2)
                        # print("Changer res")

                        if not statement:
                            self.resolution -= 0.03
                        # else: self.resolution += 0.03

                        if metric < best_metric:
                            best_metric = metric
                            self.best_delta = delta

                # Output the best sigma value and the corresponding metric
                # print("Best delta: ", self.best_delta)
                # print("Best metric: ", best_metric)

                # print("Best Metric: ", best_metric)

                period1 = int(self.peaks[i] - self.best_delta)
                period2 = int(self.peaks[i] + self.best_delta)

                popt, pcov = curve_fit(
                    f=gauss,
                    xdata=self.q[period1:period2],
                    ydata=self.difference[period1:period2],
                    bounds=(self.delta_q ** 4, [2 * 2 * self.max_I, 1, ]),
                    sigma=self.dI[period1:period2]
                )

                perr = best_metric

                self.params = np.append(self.params, self.q[self.peaks[i]])
                self.params = np.append(self.params, popt[0])
                self.params = np.append(self.params, popt[1])
                self.resolution *= RESOLUTION_FACTOR

                # plt.clf()
                # plt.plot(self.q, gauss(self.q, popt[0], popt[1]))
                # plt.plot(self.q, self.I_background_filtered)
                # plt.plot(self.q[period1:period2], self.I_background_filtered[period1:period2])
                # plt.savefig('heap/' + str(self.peak_number) + '.png')

                return gauss(self.q, popt[0], popt[1]), \
                    period1, period2, i, \
                    self.q[self.peaks[i]], \
                    gauss(self.q, popt[0], popt[1])[self.peaks[i]], popt[0], popt[1], self.peaks[i], perr
        else:
            pass

    def peak_fitting(self, i, width_factor=1):
        if np.size(self.peaks) != 0:
            delta = min(abs(self.peaks[i] - self.peaks_data['left_bases'][i]),
                        abs(self.peaks[i] - self.peaks_data['right_bases'][i]))
            # period1 = self.peaks[i] - int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
            # period2 = self.peaks[i] + int(width_factor * SIGMA_FITTING * self.peak_widths[0][i])
            period1 = self.peaks[i] - delta
            period2 = self.peaks[i] + delta

            gauss = lambda x, c, b: c * np.exp(-(x - self.q[self.peaks[i]]) ** 2 / (b ** 2))

            if period1 != period2:
                popt, pcov = curve_fit(
                    f=gauss,
                    xdata=self.q[period1:period2],
                    ydata=self.difference[period1:period2],
                    bounds=(self.delta_q ** 4, [2 * 2 * self.max_I, 1, ]),
                    sigma=self.dI[period1:period2]
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
        # self.custom_peak_fitting_with_parabole(i)
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

    def filtering_negative(self):
        self.difference = np.maximum(self.difference, 0)

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

    def state_plot(self):
        plt.clf()
        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data')
        plt.plot(self.q, self.difference_start, label='filtered_raw_data')
        plt.plot(self.q, self.difference, 'x', label='filtered_data')
        plt.plot(self.q[self.peaks], self.difference_start[self.peaks], "x", label='all_peaks_detected')
        plt.legend()

        plt.savefig(self.file_analyse_dir_peaks + '/state_plot_' + self.filename + '_peak_num:' + str(
            self.peak_number) + '.pdf')

    def result_plot(self):
        plt.clf()
        self.peaks_detected = self.peaks_detected.astype(int)

        plt.plot(self.q, self.I - BACKGROUND_COEF * self.background, linewidth=0.5, label='raw_data_without_back')
        plt.plot(self.q, self.difference_start, label='filtered_raw_data_without_back')
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
        current_peak = 0
        while len(self.peaks) > -1 and number_peak > 0:
            # self.custom_peak_searching()
            self.peak_searching(height=0, prominence=PROMINENCE, distance=6)
            if len(self.peaks) == 0:
                break
            if self.peak_verifying(current_peak):
                current_peak += 1
                # pass
            if len(self.peaks) == 0:
                break
            number_peak -= 1
            self.peak_substraction(current_peak)
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
        I_raw = self.I[self.peaks_detected][sorted_indices_q]
        dI = self.dI[self.peaks_detected][sorted_indices_q]
        peaks_detected = self.peaks_detected[sorted_indices_q]
        # print(self.params)

        return {
            'peak_number': self.peak_number,
            'q': q.tolist(),
            'I': I.tolist(),
            'dI': dI.tolist(),
            'I_raw': I_raw.tolist(),
            'peaks': peaks_detected.tolist(),
            'params_amplitude': self.params.tolist()[::3],
            'params_mean': self.params.tolist()[1::3],
            'params_sigma': self.params.tolist()[2::3],
            'start_loss': self.start_loss,
            'final_loss': self.final_loss,
            # 'loss_ratio': self.final_loss / self.start_loss
        }
