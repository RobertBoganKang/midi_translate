import numpy as np
import soundfile as sf

from sound_library.piano_1 import SoundLibrary
from sound_library.utils import ReverbExample, SoundCommon
# https://github.com/RobertBoganKang/midi_translate
from translate import MidiTranslate


class Synthesize(SoundCommon):
    """
    synthesize midi with virtual sound (for demo)
    refer to `KONTAKT` virtual instrument making method
    """

    def __init__(self, prepare_sound=True):
        # sound library
        super().__init__()
        self.piano_sustain = True
        # for longer samples buffer
        self.empty_track_end_buffer_time = 1
        self.min_trim_energy = 1e-5

        # switch
        self.reverb_add = True
        self.nonlinear_add = False

        # system parameters
        self.rvb = None
        self.sound_library = SoundLibrary()

        # if False, prepare it manually
        # usage: could change parameters before sound libs preparation
        if prepare_sound:
            self.prepare_sound_library()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.sound_library.prepare_sound_library()
        if self.reverb_add:
            self.rvb = ReverbExample()

    def create_empty_track(self, duration):
        return np.array([[0.0, 0.0] for _ in range(int(duration * self.sample_rate))])

    def trim_end_audio(self, arr):
        i = len(arr) - 1
        while i > 0:
            one_sample = arr[i]
            if one_sample[0] > self.min_trim_energy:
                break
            i -= 1
        return arr[:i + 1]

    def synthesize(self, mid_path, out_path):
        # analyze midi
        mt = MidiTranslate(mid_path)
        duration = mt.music_duration
        if self.piano_sustain:
            mt.apply_sustain_pedal_to_notes()
        notes = mt.notes
        # create performance
        synthesized_audio = self.create_empty_track(duration + self.empty_track_end_buffer_time)
        for ch, channel_notes in notes.items():
            for n in channel_notes:
                sample = self.sound_library.get_sample(n.pitch, n.velocity, n.duration)
                starting_sample = int(round(n.start * self.sample_rate))
                synthesized_audio[starting_sample:starting_sample + len(sample)] += sample
        # apply nonlinear
        if self.nonlinear_add:
            synthesized_audio = self.audio_nonlinear_transform(synthesized_audio, self.nonlinear)
        # apply reverb
        if self.reverb_add:
            synthesized_audio = self.rvb.apply(synthesized_audio, self.sample_rate)
        # norm audio
        synthesized_audio = self.norm_audio(synthesized_audio)
        synthesized_audio = self.trim_end_audio(synthesized_audio)
        # export
        if len(synthesized_audio) > 0:
            sf.write(out_path, synthesized_audio, samplerate=self.sample_rate)
        else:
            print('WARNING: audio empty!')


if __name__ == '__main__':
    import sys

    s = Synthesize()
    s.synthesize(sys.argv[1], sys.argv[2])
