import random

import numpy as np


class SoundCommon(object):
    def __init__(self):
        self.a4_frequency = 440
        self.sample_rate = 48000
        self.max_note_duration = 10

        # parameters
        self.random_sample_shift = 0.01
        self.nonlinear = 0.06
        self.highpass_freq = 20

    def key_to_frequency(self, key, shift=0):
        return self.a4_frequency * 2 ** ((key - 49) / 12 + shift / 1200)

    @staticmethod
    def norm_audio(audio):
        max_data = np.max(np.abs(audio))
        if max_data > 0:
            audio /= max_data
        return audio

    def calculate_start_idx(self, left_mic, delta_time):
        if left_mic:
            if delta_time < 0:
                start_idx = 0
            else:
                start_idx = delta_time
        else:
            if delta_time < 0:
                start_idx = -delta_time
            else:
                start_idx = 0
        return int(start_idx * self.sample_rate)

    def get_start_idx(self):
        return int(random.uniform(0, self.random_sample_shift * self.sample_rate))

    def audio_nonlinear_transform(self, arrays, nonlinear):
        from scipy.signal import butter, filtfilt
        if nonlinear == 0:
            return arrays
        nonlinear_array = (nonlinear + 1) ** ((np.array(arrays) - 1) / 2)
        nonlinear_array -= np.mean(nonlinear_array)
        nonlinear_array = self.norm_audio(nonlinear_array)
        nyq = self.sample_rate / 2
        # noinspection PyTupleAssignmentBalance
        b, a = butter(5, self.highpass_freq / nyq, btype='high', analog=False)
        # for stereo
        nonlinear_array = np.transpose(nonlinear_array)
        nonlinear_array = np.array([filtfilt(b, a, nonlinear_array[i]) for i in range(len(nonlinear_array))])
        nonlinear_array = np.transpose(nonlinear_array)
        nonlinear_array = self.norm_audio(nonlinear_array)
        return nonlinear_array


class SpatialModeling(object):
    def __init__(self):
        self.mic_source_distance = 0.4
        self.source_width = 1.2
        self.mic_distance = 0.3

        # for sound speed
        self.environment_temperature = 20

    def calculate_sound_speed(self):
        """ sound speed will change by temperature """
        return 331 * (1 + self.environment_temperature / 273) ** (1 / 2)

    def spatial_key_to_x(self, key):
        return (key - 44) / 88 * self.source_width / 2

    def get_parameter(self, key):
        source_x = self.spatial_key_to_x(key)
        left_mic_x = -self.mic_distance / 2
        right_mic_x = self.mic_distance / 2
        mic_y = self.mic_source_distance
        left_source_mic_distance = np.sqrt(np.power(source_x - left_mic_x, 2) + np.power(mic_y, 2))
        right_source_mic_distance = np.sqrt(np.power(source_x - right_mic_x, 2) + np.power(mic_y, 2))
        sound_speed = self.calculate_sound_speed()
        delta_time = (left_source_mic_distance - right_source_mic_distance) / sound_speed
        if left_source_mic_distance < right_source_mic_distance:
            sound_power_ratio = [1, np.power(left_source_mic_distance / right_source_mic_distance, 2)]
        else:
            sound_power_ratio = [np.power(right_source_mic_distance / left_source_mic_distance, 2), 1]
        return delta_time, sound_power_ratio


class ReverbExample(object):
    """ reverb simulation (example) """

    def __init__(self):
        self.plot_room = False
        self.room_size = (12, 16)
        self.source_y = 13
        self.source_width = 0.3
        self.source_mic_position = 0.6
        self.room_material = (0.2, 0.2)

    # noinspection PyDeprecation
    def apply(self, audio, sr):
        import pyroomacoustics as pra
        corner = np.array(
            [[0, 0], [self.room_size[0], 0], [self.room_size[0], self.room_size[1]], [0, self.room_size[1]]]).T
        room = pra.room.Room.from_corners(corner, fs=sr,
                                          max_order=3,
                                          materials=pra.Material(*self.room_material),
                                          ray_tracing=True, air_absorption=True)
        room.add_source([self.room_size[0] / 2 - self.source_width / 2, self.source_y], signal=audio.T[0])
        room.add_source([self.room_size[0] / 2 + self.source_width / 2, self.source_y], signal=audio.T[1])
        # add microphone
        rr = pra.circular_2D_array(center=[self.room_size[0] / 2, self.source_y - self.source_mic_position], M=2,
                                   phi0=0, radius=1e-3)
        room.add_microphone_array(pra.MicrophoneArray(rr, room.fs))
        room.simulate()
        sim_audio = room.mic_array.signals.T

        if self.plot_room:
            fig, ax = room.plot()
            ax.set_xlim([-1, self.room_size[0] + 1])
            ax.set_ylim([-1, self.room_size[1] + 1])
            fig.show()
        return sim_audio
