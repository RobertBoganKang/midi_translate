import os
import sys

import numpy as np

sys.path.append(os.path.relpath(os.path.dirname(__file__)))
try:
    from utils import SoundCommon, SpatialModeling
except ImportError:
    raise 'ERROR: cannot import!'


class Percussion(SoundCommon):
    """
    ref: https://www.youtube.com/watch?v=ogFAHvYatWs&t=254s
    """
    def __init__(self):
        super().__init__()
        # define ending caps
        self.sample_fix_ending_time = 0.06
        self.max_note_duration = 1

        # system parameters
        self.sample_fix_ending = None
        self.percussion_sound_0 = None
        self.percussion_sound_1 = None

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.build_fix_caps()
        self.build_sound_library()

    def build_fix_caps(self):
        ending_sample = self.sample_fix_ending_time * self.sample_rate
        self.sample_fix_ending = np.blackman(int(ending_sample * 2))[-int(ending_sample):]

    def build_sound_library(self):
        self.percussion_sound_0 = self.create_sample_0()
        self.percussion_sound_1 = self.create_sample_1()

    def create_sample_0(self):
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        audio = np.clip(1.5 * np.sin(2000 * np.exp(-15 * x) * x), -1, 1)
        audio[-len(self.sample_fix_ending):] *= self.sample_fix_ending
        return audio

    def create_sample_1(self):
        x = np.arange(0, self.max_note_duration, 1 / self.sample_rate)
        audio = np.clip(1.5 * np.sin(4000 * np.exp(-25 * x) * x), -1, 1)
        audio[-len(self.sample_fix_ending):] *= self.sample_fix_ending
        return audio

    def get_sample(self, beat_in_bar, tempo_numerator):
        den, velocity = self.percussion_velocity(beat_in_bar, tempo_numerator)
        if den == 1:
            sample = velocity * self.percussion_sound_0
        else:
            sample = velocity * self.percussion_sound_1
        sample = np.transpose([sample, sample])
        return sample
