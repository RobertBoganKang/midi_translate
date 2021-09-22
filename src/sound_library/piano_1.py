import os
import random
import sys

import numpy as np

sys.path.append(os.path.relpath(os.path.dirname(__file__)))
try:
    from utils import SoundCommon, SpatialModeling
except ImportError:
    raise 'ERROR: cannot import!'


class SoundLibrary(SoundCommon):
    def __init__(self):
        super().__init__()
        # define heading/ending caps
        self.sample_heading_time = 0.006
        self.piano_max_ending_sample_time = 0.3
        self.piano_max_ending_sample_power_param = 20
        self.piano_max_ending_sample_pre_time = 0.1

        # piano sound
        self.piano_final_energy = 0.01
        self.piano_volume_decay_param = 0.8
        self.piano_overtone_decay_param = 0.1
        self.piano_hard_sound_power_param = 0.7
        self.piano_warping_power_param = 3
        self.piano_max_f0_volume = 0.37
        self.piano_mix_f0_param_0 = 0.6
        self.piano_mix_f0_param_1 = 0.006

        # bad temperment variations
        self.random_temperment = 0.5
        self.random_unison = 0.7

        # system parameters
        self.sound_library_sin_1 = {}
        self.sound_library_sin_2 = {}
        self.unison_strings = {}
        self.sample_heading = None

        # spatial modeling
        self.spatial_modeling = SpatialModeling()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.get_unison_string_info()
        self.build_heading_caps()
        self.build_sound_library()

    def get_unison_string_info(self):
        for key in range(1, 89):
            if key < 10:
                self.unison_strings[key] = 1
            elif key < 32:
                self.unison_strings[key] = 2
            else:
                self.unison_strings[key] = 3

    def build_sound_library(self):
        for key in range(1, 89):
            random_temperment = random.uniform(-self.random_temperment, self.random_temperment)
            sin_1_samples = []
            sin_2_samples = []
            for strings in range(self.unison_strings[key]):
                random_unison = random.uniform(-self.random_unison, self.random_unison)
                random_shift = random_temperment + random_unison
                sin_1_samples.append(self.create_sin_sample(key, random_shift, factor=1))
                sin_2_samples.append(self.create_sin_sample(key, random_shift, factor=2))
            self.sound_library_sin_1[key] = np.mean(sin_1_samples, axis=0)
            self.sound_library_sin_2[key] = np.mean(sin_2_samples, axis=0)

    def build_heading_caps(self):
        heading_sample = self.sample_heading_time * self.sample_rate
        self.sample_heading = np.blackman(int(heading_sample * 2))[:int(heading_sample)]

    def build_dynamic_end_caps(self, pitch):
        ending_sample = self.piano_max_ending_sample_time * self.piano_max_ending_sample_power_param ** (
                -pitch / 88)
        ending_sample *= self.sample_rate
        sample_ending = np.hanning(int(ending_sample * 2))[-int(ending_sample):]
        return sample_ending

    def create_sin_sample(self, key, random_shift, factor=1):
        frequency = self.key_to_frequency(key, random_shift) * factor
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        audio = np.cos(frequency * 2 * np.pi * x)
        return audio

    def get_piano_volume_decay(self, key, velocity, length):
        x = np.array([i / self.sample_rate for i in range(length)])
        volume = (1 - self.piano_final_energy) * (
                2 + self.piano_volume_decay_param * key * velocity * (key / 88) ** 2) ** (-x) + self.piano_final_energy
        overtone_volume = (1 + self.piano_overtone_decay_param * key * velocity * (key / 88) ** 2) ** (-x)
        return volume, overtone_volume

    def get_piano_f0_mix_curve(self, key):
        """ increase low key overtone and reduce high key overtone """
        return self.piano_max_f0_volume * np.power(2, (-self.piano_mix_f0_param_1 * key))

    def get_sample_single_mic(self, key, velocity, random_sample_shift_idx, sample_length, sample_ending,
                              delta_time, left_mic=True):
        """
        ref: https://www.youtube.com/watch?v=ogFAHvYatWs
        """
        start_idx = self.calculate_start_idx(left_mic, delta_time)
        random_sample_shift_idx += start_idx
        sin_sample_1 = self.sound_library_sin_1[key][
                       random_sample_shift_idx:sample_length + random_sample_shift_idx]
        sin_sample_2 = self.sound_library_sin_2[key][
                       random_sample_shift_idx:sample_length + random_sample_shift_idx]
        # mix sound
        decay_volume, overtone_decay_volume = self.get_piano_volume_decay(key, velocity, sample_length)
        final_decay_volume = overtone_decay_volume * np.power(velocity, self.piano_hard_sound_power_param)
        # trim sample
        valid_sample_length = min(len(final_decay_volume), len(sin_sample_1), len(sin_sample_2))
        final_decay_volume = final_decay_volume[:valid_sample_length]
        sin_sample_1 = sin_sample_1[:valid_sample_length]
        sin_sample_2 = sin_sample_2[:valid_sample_length]
        decay_volume = decay_volume[:valid_sample_length]
        # mix sample
        piano_f0_volume = self.get_piano_f0_mix_curve(key)
        final_decay_volume *= piano_f0_volume
        sample = sin_sample_1 * (1 - final_decay_volume) + sin_sample_2 * final_decay_volume
        sample = self.piano_mix_f0_param_0 * sample + (1 - self.piano_mix_f0_param_0) * np.power(
            sample, self.piano_warping_power_param)
        # adding caps
        sample[:len(self.sample_heading)] *= self.sample_heading
        sample[-len(sample_ending):] *= sample_ending
        # apply velocity and volume decay
        sample = self.norm_audio(sample) * decay_volume * velocity
        return sample

    def get_sample(self, key, velocity, note_duration):
        velocity /= 127
        # shift sample for phase variation
        random_sample_shift_idx = self.get_start_idx()
        sample_ending = self.build_dynamic_end_caps(key)
        sample_length = int(round(note_duration * self.sample_rate) + len(sample_ending)
                            - self.piano_max_ending_sample_pre_time)
        sample_length = max(sample_length, len(self.sample_heading) + len(sample_ending))
        delta_time, sound_power_ratio = self.spatial_modeling.get_parameter(key)
        left_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                 sample_ending, delta_time, left_mic=True) * sound_power_ratio[0]
        right_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                  sample_ending, delta_time, left_mic=True) * sound_power_ratio[1]
        sample = np.transpose([left_sample, right_sample])
        return sample
