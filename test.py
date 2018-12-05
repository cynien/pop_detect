import unittest
import audio_in
import time
import numpy as np
from scipy import signal

"""
class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
"""


class TestAudioInStream(unittest.TestCase):
    def test_input_data_size(self):
        process = audio_in.AudioInProcess()
        process.start()
        while 1:
            time.sleep(0.1)
            break
        data_size = np.shape(process.data)[0]
        self.assertEqual(data_size, 2 * process.input_audiobufferSize)

    def test_data_size(self):
        process = audio_in.AudioInProcess()
        data_size = np.shape(process.data)[0]
        self.assertEqual(data_size, 2 * process.input_audiobufferSize)

    def test_data_tail_size(self):
        process = audio_in.AudioInProcess()
        data_tail_size = np.shape(process.data_tail)[0]
        self.assertEqual(data_tail_size, 2 * process.single_data_tail_size)

    def test_analy_buffer_l_size(self):
        process = audio_in.AudioInProcess()
        buffer_size = np.shape(process.analy_buffer_l)[0]
        buffer_size_define = process.input_audiobufferSize + \
                             process.single_data_tail_size
        self.assertEqual(buffer_size, buffer_size_define)

    def test_analy_buffer_r_size(self):
        process = audio_in.AudioInProcess()
        buffer_size = np.shape(process.analy_buffer_r)[0]
        buffer_size_define = process.input_audiobufferSize + \
                             process.single_data_tail_size
        self.assertEqual(buffer_size, buffer_size_define)

    def test_keep_data_tail(self):
        process = audio_in.AudioInProcess()
        process.data.fill(1)
        process.keep_data_tail()
        self.assertEqual(np.sum(process.data_tail),
                         np.shape(process.data_tail)[0])

    def test_fill_analy_buffer_l(self):
        process = audio_in.AudioInProcess()
        process.data.fill(1)
        process.data_tail.fill(2)
        process.fill_analy_buffer()
        amount_1 = 0
        amount_2 = 0
        amount_other = 0
        for elm in process.analy_buffer_l:
            if elm == 1:
                amount_1 += 1
            elif elm == 2:
                amount_2 += 1
            else:
                amount_other += 1
        self.assertEqual([amount_1, amount_2, amount_other],
                         [np.shape(process.data)[0] / 2,
                          np.shape(process.data_tail)[0] / 2,
                          0])

    def test_fill_analy_buffer_r(self):
        process = audio_in.AudioInProcess()
        process.data.fill(1)
        process.data_tail.fill(2)
        process.fill_analy_buffer()
        amount_1 = 0
        amount_2 = 0
        amount_other = 0
        for elm in process.analy_buffer_r:
            if elm == 1:
                amount_1 += 1
            elif elm == 2:
                amount_2 += 1
            else:
                amount_other += 1
        self.assertEqual([amount_1, amount_2, amount_other],
                         [np.shape(process.data)[0] / 2,
                          np.shape(process.data_tail)[0] / 2,
                          0])

    def test_smooth_buffer_size(self):
        process = audio_in.AudioInProcess()
        buffer_size = [np.shape(process.smooth_buffer_l)[0],
                       np.shape(process.smooth_buffer_r)[0]]
        process.smooth_data()
        data_size_after_conv = [np.shape(process.smooth_buffer_l)[0],
                                np.shape(process.smooth_buffer_r)[0]]
        self.assertEqual(buffer_size, data_size_after_conv)

    def test_stream_start_pause_stop(self):
        process = audio_in.AudioInProcess()
        start_data = pause_data = restart_data = []
        process.start()
        count = 0
        while 1:
            count += 1
            time.sleep(.2)
            if count == 2:
                start_data = process.data[:]
                print("start_data", start_data)
            elif count == 4:
                process.pause()
                pause_data = process.data[:]
                print("pause_data", pause_data)
            elif count == 5:
                process.start()
            elif count == 6:
                restart_data = process.data[:]
                print("restart_data", restart_data)
            elif count == 7:
                process.close()
                break
        check_pause = np.array_equal(start_data, pause_data)
        print("check_pause", check_pause)
        check_restart = np.array_equal(pause_data, restart_data)
        self.assertEqual([check_pause, check_restart], [False, False])


class TestAudioInSignalProcess(unittest.TestCase):
    def create_sample_triangle(self, sample_size):
        amp1 = 32000
        period1 = int(2 * 20)
        amp2 = 24000
        period2 = int(2 * 30)

        def signal_function(x, amp, period):
            helf_period = int(period / 2)
            if x % period <= helf_period:
                return -amp + 2 * amp / helf_period * (x % period)
            else:
                return -amp + 2 * amp / helf_period * (period - (x % period))

        signal_l = np.array([signal_function(x, amp1, period1)
                             for x in range(sample_size)], dtype=np.int16)
        signal_r = np.array([signal_function(x, amp2, period2)
                             for x in range(sample_size)], dtype=np.int16)

        signal = np.zeros(2 * sample_size, dtype=np.int16)
        signal[::2] = signal_l
        signal[1::2] = signal_r
        return signal

    def signal_to_frames(self, frame_size, signal):
        frame_size *= 2
        frames = []
        idx_frame = 0
        while len(signal) > frame_size * idx_frame:
            idx_start = idx_frame * frame_size
            idx_end = (idx_frame + 1) * frame_size
            frames.append(signal[idx_start: idx_end])
            idx_frame += 1
        return frames

    def fill_analy_buffer(self):
        signal = self.create_sample_triangle(10000)
        frames = self.signal_to_frames(signal)
        process = audio_in.AudioInProcess()
        process.data = frames[0][:]
        process.keep_data_tail()
        process.data = frames[1][:]
        process.fill_analy_buffer()

    def compare_signal_equal(self, target_signal_stero_buffer, base_signal, period):
        """
        compare if the two peroidic signals are the same. Base signal must longer then
        target signal above a period. The function will try each phase of
        base signal until they are match.
        :param target_signal_stero_buffer: the period signal what to match to base signal.
        :param base_signal: the base signal.
        :return: True if they are the same.
        """
        len_target = np.shape(target_signal_stero_buffer[0])[0]
        target_signal = np.zeros(2 * len_target, dtype=np.int16)
        target_signal[::2] = target_signal_stero_buffer[0][:]
        target_signal[1::2] = target_signal_stero_buffer[1][:]
        len_target = np.shape(target_signal)[0]
        for phase in range(period):
            if base_signal[phase] == target_signal[0]:
                if np.array_equal(base_signal[phase:phase + len_target], target_signal):
                    return True
        return False

    def create_a_stero_series(self):
        series_l = np.array(
            [205381.07431861453, 196836.6585679873, 246306.50627095863,
             262965.84876750055, 266836.59825951443, 220731.28160025884,
             219550.01986458097, 196737.75213607468, 224188.79459877528,
             201679.2141486654, 233622.26771838518, 201472.38816473083,
             237887.0040335832, 202393.6389164342, 182463.03387059408,
             257367.96514542, 253274.7397172369, 219590.36492272635,
             267360.8584064249, 267930.6220901662, 215697.95610096902,
             224279.66022331855, 260246.6568775958])
        series_r = np.array(
            [174040.2729943956, 184643.25706341738, 197984.73256051302, 188618.60248559553,
             192597.4512733097, 152983.65347014304, 193972.9242875477, 191827.6238050854,
             177232.79062998405, 209308.45101941502, 151712.4550569536, 147662.72910652976,
             171767.94261744947, 191531.7746353238, 202568.70618142484, 140991.8299992573])
        return [series_l, series_r]

    def test_analy_buffer_fill_correct(self):
        process = audio_in.AudioInProcess()
        signal = self.create_sample_triangle(10000)
        frames = self.signal_to_frames(process.input_audiobufferSize,
                                       signal)
        process.data = frames[0][:]
        process.keep_data_tail()
        process.data = frames[1][:]
        process.fill_analy_buffer()
        ana_buffer = [process.analy_buffer_l[:], process.analy_buffer_r[:]]
        cp_result = self.compare_signal_equal(ana_buffer, signal, 600)
        self.assertEqual(cp_result, True)

    def test_find_peak_position(self):
        points = 500
        process = audio_in.AudioInProcess()
        signal = self.create_sample_triangle(points)
        stero_data = [signal[::2], signal[1::2]]
        peak_pos = process.find_peak_positions(stero_data)
        wanted_pos = [np.array(range(20, points - 8, 20)),
                      np.array(range(30, points - 8, 30))]
        result = [np.array_equal(peak_pos[0], wanted_pos[0]),
                  np.array_equal(peak_pos[1], wanted_pos[1])]
        self.assertEqual(result, [True, True])

    def test_find_helf_cycle_peroids(self):
        points = 500
        process = audio_in.AudioInProcess()
        signal = self.create_sample_triangle(points)
        stero_data = [signal[::2], signal[1::2]]
        peak_pos = process.find_peak_positions(stero_data)
        helf_cycle_period = process.find_helf_cycle_peroids(peak_pos)
        lens = [np.shape(x)[0] for x in helf_cycle_period]
        wanted = [np.full(lens[0], 20), np.full(lens[1], 30)]
        result = [np.array_equal(helf_cycle_period[x], wanted[x]) for
                  x in range(2)]
        self.assertEqual(result, [True, True])

    def test_find_smoothed_amps(self):
        process = audio_in.AudioInProcess()
        signal = self.create_sample_triangle(10000)
        frames = self.signal_to_frames(process.input_audiobufferSize,
                                       signal)
        process.data = frames[0][:]
        process.keep_data_tail()
        process.data = frames[1][:]
        process.fill_analy_buffer()
        process.smooth_data()
        peak_pos = process.find_peak_positions([process.smooth_buffer_l,
                                                process.smooth_buffer_r])
        peaks = process.find_smoothed_amps(process.smooth_buffer_l,
                                           process.smooth_buffer_r,
                                           peak_pos)
        print(2 * np.max(process.smooth_buffer_l), 2 * np.max(process.smooth_buffer_r))
        peaks_l = np.full(np.shape(peak_pos[0])[0] - 1,
                          2 * np.max(process.smooth_buffer_l))
        peaks_r = np.full(np.shape(peak_pos[1])[0] - 1,
                          2 * np.max(process.smooth_buffer_r))
        wanted = [peaks_l, peaks_r]
        result = [np.array_equal(peaks[x], wanted[x]) for
                  x in range(2)]
        self.assertEqual(result, [True, True])

    def test_cale_freq_factor(self):
        sample_rate = 44100
        process = audio_in.AudioInProcess()
        process.sampleRate = sample_rate
        process.cale_freq_factor()
        result = process.freq_factor
        self.assertEqual(result, sample_rate / 2)

    def test_cal_amp_factor(self):
        process = audio_in.AudioInProcess()
        process.filter_order = 9
        process.filter = signal.gaussian(process.filter_order, std=1.59)
        process.cale_amp_factor()
        result = process.amp_factor
        base = np.sum(signal.gaussian(9, std=1.59))
        self.assertEqual(result, base)

    def test_find_mid_avgs(self):
        stero_series = self.create_a_stero_series()
        process = audio_in.AudioInProcess()
        sorted_stero_series = process.sorted_stero_series(stero_series)
        avgs = process.find_mid_avgs(sorted_stero_series)
        self.assertEqual(avgs, [224482.47381269286, 185506.60620358019])

    def test_real_freq(self):
        process = audio_in.AudioInProcess()
        sample_rate = 44100
        helf_periods = [20, 30]
        process.sampleRate = sample_rate
        process.cale_freq_factor()
        freqs = process.find_real_freqs(helf_periods)
        base = [sample_rate / 2 / period for period in helf_periods]
        result = np.array_equal(freqs, base)
        self.assertTrue(result)

    def test_real_vpps(self):
        process = audio_in.AudioInProcess()
        amp_factor = 10
        process.amp_factor = amp_factor
        smooth_vpps = [2000, 3000]
        vpps = process.find_real_vpps(smooth_vpps)
        base = [smooth_vpp / amp_factor for smooth_vpp in smooth_vpps]
        result = np.array_equal(vpps, base)
        self.assertTrue(result)


class TestAudioInPopCheck(unittest.TestCase):
    def create_a_stero_series(self):
        series_l = np.array(
            [205381.07431861453, 196836.6585679873, 246306.50627095863,
             262965.84876750055, 266836.59825951443, 220731.28160025884,
             219550.01986458097, 196737.75213607468, 224188.79459877528,
             201679.2141486654, 233622.26771838518, 201472.38816473083,
             237887.0040335832, 202393.6389164342, 182463.03387059408,
             257367.96514542, 253274.7397172369, 219590.36492272635,
             267360.8584064249, 267930.6220901662, 215697.95610096902,
             224279.66022331855, 260246.6568775958])
        series_r = np.array(
            [174040.2729943956, 184643.25706341738, 197984.73256051302, 188618.60248559553,
             192597.4512733097, 152983.65347014304, 193972.9242875477, 191827.6238050854,
             177232.79062998405, 209308.45101941502, 151712.4550569536, 147662.72910652976,
             171767.94261744947, 191531.7746353238, 202568.70618142484, 140991.8299992573])
        return [series_l, series_r]

    def create_a_mono_series(self):
        series_l = np.array(
            [205381.07431861453, 196836.6585679873, 246306.50627095863,
             262965.84876750055, 266836.59825951443, 220731.28160025884,
             219550.01986458097, 196737.75213607468, 224188.79459877528,
             201679.2141486654, 233622.26771838518, 201472.38816473083,
             237887.0040335832, 202393.6389164342, 182463.03387059408,
             257367.96514542, 253274.7397172369, 219590.36492272635,
             267360.8584064249, 267930.6220901662, 215697.95610096902,
             224279.66022331855, 260246.6568775958])
        return [series_l]

    def test_check_series_pass(self):
        stero_series = self.create_a_stero_series()
        process = audio_in.AudioInProcess()
        sorted_stero_series = process.sorted_stero_series(stero_series)
        print(sorted_stero_series)
        avgs = process.find_mid_avgs(sorted_stero_series)
        check_pass = process.check_series_pass(sorted_stero_series, avgs, 0.22)
        self.assertFalse(check_pass)

    def test_check_series_pass_1ch(self):
        stero_series = self.create_a_stero_series()
        process = audio_in.AudioInProcess()
        sorted_stero_series = process.sorted_stero_series(stero_series)
        print(sorted_stero_series)
        avgs = process.find_mid_avgs(sorted_stero_series)
        print("avgs,", avgs)
        print("sorted_seeries[0][0]", sorted_stero_series[0][0])
        print("sorted_seeries[0][-1]", sorted_stero_series[0][-1])
        check_pass = process.check_series_pass(sorted_stero_series, avgs, 0.17)
        self.assertFalse(check_pass)

    def test_update_no_ng_time(self):
        process = audio_in.AudioInProcess()

        def assert_no_ng_time_result(time, time_last):
            delta_almost = 0.02
            print(process.no_ng_time, process.no_ng_time_last, time, time_last)
            print("compare no_ng_time")
            self.assertAlmostEqual(process.no_ng_time, time, delta=delta_almost)
            print("compare no_ng_time_last")
            self.assertAlmostEqual(process.no_ng_time_last, time_last, delta=delta_almost)

        def set_delay_time(event_time, accumulate_time):
            delay_time = event_time - accumulate_time
            print("delay_time, accumulate_time", delay_time, accumulate_time)
            return delay_time, event_time

        # time start
        accumulated_time = 0
        process.pop_test_pass = False
        # test init
        process.update_no_ng_time()
        assert_no_ng_time_result(0, 0)
        # test start no pop
        process.pop_test_pass = True
        process.update_no_ng_time()
        assert_no_ng_time_result(0, 0)
        # test 1s no_ng
        process.pop_test_pass = True
        delay_time, accumulated_time = set_delay_time(1, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(1, 0)
        # test 2.5s no_ng
        process.pop_test_pass = True
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(2.5, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(2.5, 0)
        # test 3s ng
        process.pop_test_pass = False
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(3, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(0, 2.5)
        # test 4s no_ng
        process.pop_test_pass = True
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(4, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(1, 2.5)
        # test 4.5s ng
        process.pop_test_pass = False
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(4.5, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(0, 2.5)
        # test 5 no_ng
        process.pop_test_pass = True
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(5, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(0.5, 2.5)
        # test 7.5 no_ng
        process.pop_test_pass = True
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(7.5, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(3, 2.5)
        # test 8s ng
        process.pop_test_pass = False
        print("befor set delay time, accumulated=", accumulated_time)
        delay_time, accumulated_time = set_delay_time(8, accumulated_time)
        time.sleep(delay_time)
        process.update_no_ng_time()
        assert_no_ng_time_result(0, 3)


if __name__ == '__main__':
    unittest.main()
