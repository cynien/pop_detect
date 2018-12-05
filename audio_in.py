import pyaudio
import numpy as np
from scipy import signal
import time
import wave
import struct


class AudioInProcess:
    """
    audio in and related sinad pop detector process.
    """

    def __init__(self, device=None):
        # pyaudio related
        self.call_back_status = pyaudio.paContinue
        self.sampleRate = 44100
        self.input_audiobufferSize = 4096  # numbers of samples
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=pyaudio.paInt16, channels=2,
                                  rate=self.sampleRate, input=True,
                                  frames_per_buffer=self.input_audiobufferSize,
                                  stream_callback=self.audio_callback, start=False)
        self.data = np.zeros(2 * self.input_audiobufferSize, 'int16')

        # smooth function's related
        self.filter_order_factor = 4
        self.filter_order = 2 * self.filter_order_factor + 1
        self.filter = signal.gaussian(self.filter_order, std=1.59)

        # tail size must greather than helf cycle points+filter_order_factor,
        # for samle rate 44.1k/s, 1KHz helf cycle period around 30 point.
        self.single_data_tail_size = 30  # per channel
        self.data_tail = np.zeros(self.single_data_tail_size * 2, 'int16')
        self.analy_buffer_size = int(self.input_audiobufferSize +
                                     self.single_data_tail_size)
        self.analy_buffer_l = np.zeros(self.analy_buffer_size, 'int16')
        self.analy_buffer_r = np.zeros(self.analy_buffer_size, 'int16')
        self.smooth_buffer_l = np.zeros(self.analy_buffer_size -
                                        self.filter_order + 1, 'int16')
        self.smooth_buffer_r = np.zeros(self.analy_buffer_size -
                                        self.filter_order + 1, 'int16')

        # judge error signal related
        self.trim_sample_rate = 0.42
        self.err_percent = 0.12  # correct period, vpp deviation limit
        self.pop_test_pass = False  # final pop test
        self.time_last_pop = time.time()
        self.no_ng_time = 0
        self.no_ng_time_last = 0

        # calculte real freq., vpp
        self.amp_factor = np.sum(self.filter)
        self.freq_factor = self.sampleRate / 2.0
        self.freq_avgs = [1000, 1000]  # final freq_avgs
        self.amp_avgs = [1000, 1000]  # final amplltude avgs

        # caculate precise no_ng_time
        self.accu_no_pop = 0  # no pop accumulate samples
        self.pops_in_buffer = []

    def start(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def close(self):
        self.set_call_back_continue(False)
        self.stream.close()
        self.p.terminate()

    def get_audio_data(self):
        """get a single buffer size worth of audio."""
        audio_string = self.stream.read(self.input_audiobufferSize)
        return np.fromstring(audio_string, dtype=np.int16)

    def read_file_stream(self):
        audio_piece = self.wf.readframes(self.input_audiobufferSize)
        if audio_piece == '':
            return False
        self.data = np.array(struct.unpack("%dh" % (2 * self.input_audiobufferSize),
                                           audio_piece))
        return True

    def update_plot(self):
        pass

    def audio_callback(self, data, frame_count, time_info, status):
        self.data = np.fromstring(data, 'int16')
        self.fill_analy_buffer()
        self.keep_data_tail()
        self.smooth_data()

        peak_positions = self.find_peak_positions([self.smooth_buffer_l,
                                                   self.smooth_buffer_r])
        helf_periods = self.find_helf_cycle_peroids(peak_positions)

        sorted_helf_periods = self.sorted_stero_series(helf_periods)
        self.helf_period_avgs = self.find_mid_avgs(sorted_helf_periods)
        self.smoothed_amps = self.find_smoothed_amps(self.smooth_buffer_l,
                                                     self.smooth_buffer_r,
                                                     peak_positions)
        sorted_smoothed_amps = self.sorted_stero_series(self.smoothed_amps)
        smoothed_amp_avgs = self.find_mid_avgs(sorted_smoothed_amps)

        # check pop pass or fail
        self.pop_test_pass = self.check_series_pass(sorted_smoothed_amps,
                                                    smoothed_amp_avgs,
                                                    self.err_percent)
        if self.pop_test_pass:
            self.pop_test_pass = self.check_series_pass(sorted_helf_periods,
                                                        self.helf_period_avgs,
                                                        self.err_percent)
        self.update_no_ng_time()

        # real freq and amplitude
        self.freq_avgs = self.find_real_freqs(self.helf_period_avgs)
        self.amp_avgs = self.find_real_vpps(smoothed_amp_avgs)

        self.update_gui()
        self.update_waveform()

        return None, self.call_back_status

    def set_call_back_continue(self, is_continue):
        """
        the function is useless, for after set paComplete,
        the callback function will not continue again
        :param is_continue: True if continue
        :return: None
        """
        if is_continue:
            self.call_back_status = pyaudio.paContinue
        else:
            self.call_back_status = pyaudio.paComplete

    def update_gui(self):
        pass

    def update_waveform(self):
        pass

    def keep_data_tail(self):
        self.data_tail = self.data[-2 * self.single_data_tail_size:]

    def fill_analy_buffer(self):
        self.analy_buffer_l[0:self.single_data_tail_size] = self.data_tail[::2]
        self.analy_buffer_l[self.single_data_tail_size:] = self.data[::2]
        self.analy_buffer_r[0:self.single_data_tail_size] = self.data_tail[1::2]
        self.analy_buffer_r[self.single_data_tail_size:] = self.data[1::2]

    def smooth_data(self):
        self.smooth_buffer_l = np.convolve(self.analy_buffer_l, self.filter, 'vaild')
        self.smooth_buffer_r = np.convolve(self.analy_buffer_r, self.filter, 'vaild')

    def find_peak_positions(self, stero_data):
        pos_buffer = [[], []]
        for n, data in enumerate(stero_data):
            pos_buffer[n] = np.greater_equal(np.diff(data), 0)
            pos_buffer[n] = np.logical_xor(pos_buffer[n][1:],
                                           pos_buffer[n][:-1])
            pos_buffer[n] = np.where(pos_buffer[n] == True)[0]
            pos_buffer[n] = np.add(pos_buffer[n], 1)
        return pos_buffer

    def find_helf_cycle_peroids(self, peak_positions):
        normalize_helf_periods = [[], []]
        for n, data in enumerate(peak_positions):
            normalize_helf_periods[n] = np.diff(data)
        return normalize_helf_periods

    def cale_freq_factor(self):
        # real freq = (self.sampleRate/2) / helf_descrate_period
        self.freq_factor = self.sampleRate / 2.0

    def find_smoothed_amps(self, data_l, data_r, peak_positions):
        amps = np.array([data_l[x] for x in peak_positions[0]]), \
               np.array([data_r[x] for x in peak_positions[1]])
        # return [np.abs(x[1:] - x[:-1]) for x in amps]
        return [np.abs(np.diff(x)) for x in amps]

    def cale_amp_factor(self):
        # real Vpp = (smoothed_vpp)/self.amp_factor
        self.amp_factor = np.sum(self.filter)

    def sorted_stero_series(self, stero_series):
        return [np.sort(x) for x in stero_series]

    def find_mid_avgs(self, sorted_stero_series):
        avgs = []
        for elm in sorted_stero_series:
            mid = int(self.trim_sample_rate * np.shape(elm)[0])
            avg = np.average(elm[mid:-mid])
            avgs.append(avg)
        return avgs

    def find_real_freqs(self, helf_period_avgs):
        return [(self.freq_factor / helf_period) for helf_period in helf_period_avgs]

    def find_real_vpps(self, smooth_amp_avgs):
        return [(smooth_vpp / self.amp_factor) for smooth_vpp in smooth_amp_avgs]
        # real Vpp = (smoothed_vpp)/self.amp_factor

    def check_series_pass(self, sorted_stero_series, avgs, fail_percent):
        # print(sorted_stero_series)
        for idx, series in enumerate(sorted_stero_series):
            if series[0] <= (1 - fail_percent) * avgs[idx]:
                return False
            elif series[-1] >= (1 + fail_percent) * avgs[idx]:
                return False
        return True

    def update_no_ng_time(self):
        time_now = time.time()
        if self.pop_test_pass:
            self.no_ng_time = time_now - self.time_last_pop
        else:
            self.time_last_pop = time_now
            if self.no_ng_time >= 1.5:
                self.no_ng_time_last = self.no_ng_time
            self.no_ng_time = 0

    def len_trim_outrange_peaks(self, peak_positions):
        trim_len = [0, 0]
        start_idx = self.single_data_tail_size - self.filter_order_factor
        for idx, position_series in enumerate(peak_positions):
            for position in position_series:
                if position < start_idx:
                    trim_len[idx] += 1
                else:
                    break
        return trim_len

    def peak_positions_shifted_for_time_count(self, trimed_peak_postions):
        shifted_peak_positions = [[], []]
        for idx, series in enumerate(trimed_peak_postions):
            shifted_peak_positions[idx] = \
                np.subtract(trimed_peak_postions,
                            self.single_data_tail_size - self.filter_order_factor)
        return shifted_peak_positions

    def map_series_postions_hperiods_amps(
            self, peak_positions, h_periods, amps, trim_len):
        trimed_peak_positions = [peak_positions[0][trim_len[0]:],
                                 peak_positions[1][trim_len[1]:]]
        shift_peak_positions = \
            self.peak_positions_shifted_for_time_count(trimed_peak_positions)
        shift_hperiod = [h_periods[0][trim_len[0]-1:],
                         h_periods[1][trim_len[1]-1:]]
        shift_amps = [amps[0][trim_len[0]-1:],
                      amps[1][trim_len[1]-1:]]
        return {"peak_pos": shift_peak_positions,
                "h_period": shift_hperiod,
                "amp": shift_amps}

    def check_series_pass_preicse(self, maped_datas, fail_percent):
        pop_peaks = [[], []]
        sorted_amps = self.sorted_stero_series(maped_datas.peak_pos)
        amps_avgs = self.find_mid_avgs(sorted_amps)
        sorted_h_periods = self.sorted_stero_series(maped_datas.h_period)
        h_period_avgs = self.find_mid_avgs(sorted_h_periods)
        for idx_lr, peak_data in enumerate(maped_datas.peak_pos):
            for idx, peak_pos in enumerate(peak_data):
                if maped_datas.amp[idx] <= (1 - fail_percent) * amps_avgs[idx]:
                    pop_peaks[idx].append(peak_pos[idx])
                elif maped_datas.amp[idx] >= (1 + fail_percent) * amps_avgs[idx]:
                    pop_peaks[idx].append(peak_pos[idx])
                elif maped_datas.h_period[idx] <= (1 - fail_percent) * h_period_avgs[idx]:
                    pop_peaks[idx].append(peak_pos[idx])
                elif maped_datas.h_period[idx] >= (1 + fail_percent) * h_period_avgs[idx]:
                    pop_peaks[idx].append(peak_pos[idx])
        return pop_peaks

    def cal_pop_times(self, pop_peaks):
        pass

    def remove_space_less_x(self, series_pop_total, x):
        sorted_series = sorted(series_pop_total)
        no_repeated_series = []
        while len(sorted_series) > 1:
            for elm in sorted_series[1:]:
                if elm - sorted_series[0] < x:
                    sorted_series.remove(elm)
                else:
                    break
            no_repeated_series.append(sorted_series[0])
            sorted_series.remove(sorted_series[0])
        return no_repeated_series


if __name__ == "__main__":
    import time

    start_data = pause_data = restart_data = []
    process = AudioInProcess()
    process.start()
    count = 0
    while 1:
        count += 1
        print("count =", count)
        print("data under count", process.data)
        time.sleep(.2)
        if count == 2:
            print("start data", process.data)
            start_data = process.data
        if count == 3:
            process.pause()
            print("pause data", process.data)
            pause_data = process.data
            print("process pause")
            print("process pause")
        elif count == 4:
            print("process start")
            process.start()
        elif count == 5:
            print("restart_data", process.data)
            restart_data = process.data
        elif count == 7:
            print("process close")
            print("data", process.data)
            process.close()
            print("data", process.data)
            break
    print(start_data)
    print(pause_data)
    print(restart_data)
