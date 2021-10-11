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
        self.sample_fix_ending_time = 0.06

        # sound effect
        self.high_velocity_type = 'triangle'
        self.high_velocity_square_power_param = 0.5

        # bad temperment variations
        self.random_temperment = 0.2
        self.random_sample_shift = 0.01

        # system parameters
        self.sound_library_sin = {}
        self.sound_library_high_velocity = {}
        self.sample_heading = None
        self.sample_fix_ending = None

        # spatial modeling
        self.spatial_modeling = SpatialModeling()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.build_sound_library()
        self.build_fix_caps()

    def build_sound_library(self):
        for key in range(1, 89):
            random_temperment = random.uniform(-self.random_temperment, self.random_temperment)
            random_shift = random_temperment
            self.sound_library_sin[key] = self.create_sin_sample(key, random_shift)
            self.sound_library_high_velocity[key] = self.create_high_velocity_sample(key, random_shift)

    def create_sin_sample(self, key, random_shift):
        frequency = self.key_to_frequency(key, random_shift)
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        audio = np.cos(frequency * 2 * np.pi * x)
        return audio

    def create_high_velocity_sample(self, key, random_shift):
        frequency = self.key_to_frequency(key, random_shift)
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        if self.high_velocity_type == 'triangle':
            audio = np.abs(2 * np.mod(x * frequency * 2, 2) - 2) - 1
        elif self.high_velocity_type == 'square':
            y = np.cos(frequency * 2 * np.pi * x)
            audio = np.sign(y) * np.power(np.abs(y), self.high_velocity_square_power_param)
        else:
            raise ValueError('ERROR: high velocity type unknown!')
        return audio

    def build_fix_caps(self):
        heading_sample = self.sample_heading_time * self.sample_rate
        self.sample_heading = np.blackman(int(heading_sample * 2))[:int(heading_sample)]
        ending_sample = self.sample_fix_ending_time * self.sample_rate
        self.sample_fix_ending = np.blackman(int(ending_sample * 2))[-int(ending_sample):]

    def get_sample_single_mic(self, key, velocity, random_sample_shift_idx, sample_length, sample_ending,
                              delta_time, left_mic=True):
        start_idx = self.calculate_start_idx(left_mic, delta_time)
        random_sample_shift_idx += start_idx
        sin_sample = self.sound_library_sin[key][
                     random_sample_shift_idx:sample_length + random_sample_shift_idx]
        high_velocity_sample = self.sound_library_high_velocity[key][
                               random_sample_shift_idx:sample_length + random_sample_shift_idx]
        sample = sin_sample * (1 - velocity) + high_velocity_sample * velocity
        # adding caps
        sample[:len(self.sample_heading)] *= self.sample_heading
        sample[-len(sample_ending):] *= sample_ending
        sample = self.norm_audio(sample) * velocity
        return sample

    def get_sample(self, key, velocity, note_duration):
        velocity /= 127
        # shift sample for phase variation
        random_sample_shift_idx = self.get_start_idx()
        sample_ending = self.sample_fix_ending
        sample_length = int(round(note_duration * self.sample_rate))
        sample_length = max(sample_length, len(self.sample_heading) + len(sample_ending))
        delta_time, sound_power_ratio = self.spatial_modeling.piano_spatial_parameter(key)
        left_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                 sample_ending, delta_time, left_mic=True) * sound_power_ratio[0]
        right_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                  sample_ending, delta_time, left_mic=True) * sound_power_ratio[1]
        sample = self.combine_stereo_sample(left_sample, right_sample)
        return sample
