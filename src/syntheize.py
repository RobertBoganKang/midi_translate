import numpy as np
import soundfile as sf

from sound_library.beats.heart import BeatLibrary
from sound_library.notes.piano_0 import NoteLibrary
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
        # for longer samples buffer
        self.empty_track_end_buffer_time = 1
        self.min_trim_energy = 1e-5

        # switch
        self.piano_sustain = True
        self.reverb_add = True
        self.nonlinear_add = False
        self.beat_sound_mix = 0

        # system parameters
        self.rvb = None
        self.note_library = NoteLibrary()
        self.beat_library = BeatLibrary()

        # if False, prepare it manually
        # usage: could change parameters before sound libs preparation
        if prepare_sound:
            self.prepare_sound_library()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.note_library.prepare_sound_library()
        if self.beat_sound_mix > 0:
            self.beat_library.prepare_sound_library()
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
        # create performance
        # (0) creat empty track
        synthesized_audio = self.create_empty_track(duration + self.empty_track_end_buffer_time)
        # (1) patch notes
        if self.beat_sound_mix < 1:
            notes = mt.notes
            for ch, channel_notes in notes.items():
                for n in channel_notes:
                    sample = self.note_library.get_sample(n.pitch, n.velocity, n.duration)
                    sample *= (1 - self.beat_sound_mix)
                    starting_sample = int(round(n.start * self.sample_rate))
                    synthesized_audio[starting_sample:starting_sample + len(sample)] += sample
            # (2) apply reverb
            if self.reverb_add:
                synthesized_audio = self.rvb.apply(synthesized_audio, self.sample_rate)
            synthesized_audio = self.norm_audio(synthesized_audio)
        # (3) patch beats
        if self.beat_sound_mix > 0:
            beats = mt.beats
            for bt in beats:
                beat_in_bar = bt.beat_in_bar
                tempo_numerator = bt.numerator
                sample = self.beat_library.get_sample(beat_in_bar, tempo_numerator)
                sample *= self.beat_sound_mix
                starting_sample = int(round(bt.time * self.sample_rate))
                synthesized_audio[starting_sample:starting_sample + len(sample)] += sample
        # (4) apply nonlinear
        if self.nonlinear_add:
            synthesized_audio = self.audio_nonlinear_transform(synthesized_audio, self.nonlinear)
        # (5) audio post processing
        # norm audio
        synthesized_audio = self.norm_audio(synthesized_audio)
        # trim audio
        synthesized_audio = self.trim_end_audio(synthesized_audio)
        # (6) export audio
        if len(synthesized_audio) > 0:
            sf.write(out_path, synthesized_audio, samplerate=self.sample_rate)
        else:
            print('WARNING: audio empty!')


if __name__ == '__main__':
    import sys

    s = Synthesize()
    s.synthesize(sys.argv[1], sys.argv[2])
