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
        self.sample_heading_time = 0.003
        self.piano_max_ending_sample_time = 0.3
        self.piano_max_ending_sample_power_param = 20
        self.piano_max_ending_sample_pre_time = 0.05
        self.piano_note_struck_volume_ratio = 0.01
        self.piano_note_release_volume_ratio = 0.06

        # piano sound
        self.piano_sample_max_overtone = 32
        self.piano_min_sample_overtone_power_param = 1.6
        self.piano_sample_overtone_power_base = 1.008
        self.piano_final_energy = 0.02
        self.piano_final_overtone_energy = 0.1
        self.piano_volume_decay_param = 0.7
        self.piano_overtone_decay_param = 0.15
        self.piano_hard_sound_power_param = 0.7
        self.piano_max_f0_volume = 0.39
        self.piano_mix_f0_param_0 = 0.6
        self.piano_mix_f0_param_1 = 0.006

        # bad temperment variations
        self.random_temperment = 0.01
        self.random_unison = 1.2

        # tuning
        self.tuning_param = 0.0004

        # system parameters
        self.sound_library_sin_0 = {}
        self.sound_library_sin_1 = {}
        self.unison_strings = {}
        self.sample_heading = None

        # spatial modeling
        self.spatial_modeling = SpatialModeling()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.get_unison_string_info()
        self.build_heading_caps()
        self.build_sound_library()

    @staticmethod
    def get_inharmonicity_ratio(key, n):
        # ref: http://www.github.com/RobertBoganKang/piano_tuning
        # TODO: change ih parameter (example: grand piano)
        ih_k = max(1 / 13 * key - 20 / 13, -0.035 * key + 0.3)
        bk = np.exp(ih_k) / 10000
        return n * np.sqrt((1 + bk * n ** 2) / (1 + bk))

    def get_tuning_shift(self, key):
        """ simple piano tuning curve """
        return self.tuning_param * (key - 49) ** 3

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
            sin_0_samples = []
            sin_1_samples = []
            for strings in range(self.unison_strings[key]):
                random_unison = random.uniform(-self.random_unison, self.random_unison)
                random_shift = random_temperment + random_unison
                sin_0_samples.append(self.create_sin_sample(key, random_shift))
                sin_1_samples.append(self.create_overtone_sample(key, random_shift))
            self.sound_library_sin_0[key] = np.mean(sin_0_samples, axis=0)
            self.sound_library_sin_1[key] = np.mean(sin_1_samples, axis=0)

    def build_heading_caps(self):
        heading_sample = self.sample_heading_time * self.sample_rate
        self.sample_heading = np.blackman(int(heading_sample * 2))[:int(heading_sample)]

    def build_dynamic_end_caps(self, pitch):
        ending_sample = self.piano_max_ending_sample_time * self.piano_max_ending_sample_power_param ** (
                -pitch / 88)
        ending_sample *= self.sample_rate
        sample_ending = np.hanning(int(ending_sample * 2))[-int(ending_sample):]
        return sample_ending

    def create_sin_sample(self, key, random_shift):
        tuning_shift = self.get_tuning_shift(key)
        f0 = self.key_to_frequency(key, random_shift + tuning_shift)
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        audio = np.cos(f0 * 2 * np.pi * x)
        return audio

    def get_sample_overtone_power_param(self, key):
        return self.piano_min_sample_overtone_power_param * self.piano_sample_overtone_power_base ** (key - 1)

    def create_overtone_sample(self, key, random_shift):
        tuning_shift = self.get_tuning_shift(key)
        f0 = self.key_to_frequency(key, random_shift + tuning_shift)
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        piano_sample_overtone_power_param = self.get_sample_overtone_power_param(key)
        audio = np.array([0.0] * len(x))
        max_overtone = min(int(self.sample_rate / 2 / f0), self.piano_sample_max_overtone)
        for i in range(2, max_overtone + 1):
            i_ih = self.get_inharmonicity_ratio(key, i)
            sine = (np.cos(i_ih * f0 * 2 * np.pi * x) / (piano_sample_overtone_power_param ** (i - 1)))
            audio += sine
        audio = self.norm_audio(audio)
        return audio

    def get_piano_volume_decay(self, key, velocity, length):
        x = np.array([i / self.sample_rate for i in range(length)])
        volume = self.piano_final_energy + (1 - self.piano_final_energy) * (
                2 + self.piano_volume_decay_param * key * velocity * (key / 88) ** 2) ** (-x)
        overtone_volume = self.piano_final_overtone_energy + (1 - self.piano_final_overtone_energy) * (
                1 + self.piano_overtone_decay_param * key * velocity * (key / 88) ** 2) ** (-x)
        return volume, overtone_volume

    def get_piano_f0_mix_curve(self, key):
        """ increase low key overtone and reduce high key overtone """
        return self.piano_max_f0_volume * np.power(2, (-self.piano_mix_f0_param_1 * key))

    @staticmethod
    def apply_power(arr, power):
        if power % 2 == 1:
            return np.power(arr, power)
        else:
            return np.sign(arr) * np.power(np.abs(arr), power)

    def get_sample_single_mic(self, key, velocity, random_sample_shift_idx, sample_length, sample_ending,
                              delta_time, left_mic=True):
        start_idx = self.calculate_start_idx(left_mic, delta_time)
        random_sample_shift_idx += start_idx
        sin_sample_1 = self.sound_library_sin_0[key][
                       random_sample_shift_idx:sample_length + random_sample_shift_idx]
        sin_sample_2 = self.sound_library_sin_1[key][
                       random_sample_shift_idx:sample_length + random_sample_shift_idx]
        # mix sound
        decay_volume, overtone_decay_volume = self.get_piano_volume_decay(key, velocity, sample_length)
        final_overtone_decay_volume = overtone_decay_volume * np.power(velocity, self.piano_hard_sound_power_param)
        # trim sample
        valid_sample_length = min(len(final_overtone_decay_volume), len(sin_sample_1), len(sin_sample_2))
        final_overtone_decay_volume = final_overtone_decay_volume[:valid_sample_length]
        sin_sample_1 = sin_sample_1[:valid_sample_length]
        sin_sample_2 = sin_sample_2[:valid_sample_length]
        decay_volume = decay_volume[:valid_sample_length]
        # mix sample
        piano_f0_volume = self.get_piano_f0_mix_curve(key)
        final_overtone_decay_volume *= piano_f0_volume
        # add struck and release overtone
        struck_head = final_overtone_decay_volume[:len(self.sample_heading)]
        struck_head_volume = decay_volume[:len(self.sample_heading)]
        struck_head += np.clip(self.piano_note_struck_volume_ratio * struck_head_volume * self.sample_heading, None, 1)
        release_head = final_overtone_decay_volume[-len(self.sample_heading) - len(sample_ending):-len(sample_ending)]
        release_head_volume = decay_volume[-len(self.sample_heading) - len(sample_ending):-len(sample_ending)]
        release_head += np.clip(self.piano_note_release_volume_ratio * release_head_volume * self.sample_heading, None,
                                1)
        struck_end = final_overtone_decay_volume[
                   len(self.sample_heading):len(self.sample_heading) + len(sample_ending):]
        struck_end_volume = decay_volume[
                          len(self.sample_heading):len(self.sample_heading) + len(sample_ending):]
        struck_end += np.clip(self.piano_note_struck_volume_ratio * struck_end_volume * sample_ending, None, 1)
        release_end = final_overtone_decay_volume[-len(sample_ending):]
        release_end_volume = decay_volume[-len(sample_ending):]
        release_end += np.clip(self.piano_note_release_volume_ratio * release_end_volume * sample_ending, None, 1)
        sample = sin_sample_1 * (1 - final_overtone_decay_volume) + sin_sample_2 * final_overtone_decay_volume
        sample = self.piano_mix_f0_param_0 * sample + (1 - self.piano_mix_f0_param_0) * sample
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
