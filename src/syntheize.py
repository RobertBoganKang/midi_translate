import random

import numpy as np
import soundfile as sf

# https://github.com/RobertBoganKang/midi_translate
from translate import MidiTranslate


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


class Synthesize(object):
    """
    synthesize midi with virtual piano sound (for demo)
        * velocity: low -> high
        * sound wave: sine -> triangle/square
    refer to `KONTAKT` virtual instrument making method
    """

    def __init__(self, prepare_sound=True):
        # sound library
        self.a4_frequency = 440
        self.sample_rate = 48000
        self.max_note_duration = 10

        # define heading/ending caps
        self.sample_heading_time = 0.006
        self.sample_fix_ending_time = 0.06
        self.piano_max_ending_sample_time = 0.3
        self.piano_max_ending_sample_power_param = 20
        self.piano_max_ending_sample_pre_time = 0.2

        # sound effect
        self.high_velocity_type = 'triangle'
        self.high_velocity_square_power_param = 0.5
        # piano sound
        self.piano_sound = True
        self.piano_sustain = True
        self.piano_final_energy = 0.01
        self.piano_final_overtone = 0.1
        self.piano_volume_decay_param = 0.8
        self.piano_overtone_decay_param = 0.03
        self.piano_hard_sound_power_param = 0.65
        # spatial modeling
        self.spatial_modeling = SpatialModeling()

        # bad temperment variations
        self.random_temperment = 0.5
        self.random_unison = 0.7
        self.random_sample_shift = 0.01

        # apply reverb
        self.reverb_add = True

        # system parameters
        self.sound_library_sin = {}
        self.sound_library_high_velocity = {}
        self.unison_strings = {}
        self.sample_heading = None
        self.sample_fix_ending = None
        self.rvb = None

        # if False, prepare it manually
        # usage: could change parameters before sound libs preparation
        if prepare_sound:
            self.prepare_sound_library()

    def prepare_sound_library(self):
        """ prepare sound library (all samples) """
        self.get_unison_string_info()
        self.build_sound_library()
        self.build_fix_caps()
        if self.reverb_add:
            self.rvb = ReverbExample()

    def key_to_frequency(self, key, random_shift=0):
        return self.a4_frequency * 2 ** ((key - 49) / 12 + random_shift / 1200)

    @staticmethod
    def norm_audio(audio):
        max_data = np.max(np.abs(audio))
        if max_data > 0:
            audio /= max_data
        return audio

    def get_unison_string_info(self):
        for key in range(1, 89):
            if self.piano_sound:
                if key < 10:
                    self.unison_strings[key] = 1
                elif key < 32:
                    self.unison_strings[key] = 2
                else:
                    self.unison_strings[key] = 3
            else:
                self.unison_strings[key] = 1

    def build_sound_library(self):
        for key in range(1, 89):
            random_temperment = random.uniform(-self.random_temperment, self.random_temperment)
            sin_samples = []
            high_velocity_samples = []
            for strings in range(self.unison_strings[key]):
                random_unison = random.uniform(-self.random_unison, self.random_unison)
                random_shift = random_temperment + random_unison
                sin_samples.append(self.create_sin_sample(key, random_shift))
                high_velocity_samples.append(self.create_high_velocity_sample(key, random_shift))
            self.sound_library_sin[key] = np.mean(sin_samples, axis=0)
            self.sound_library_high_velocity[key] = np.mean(high_velocity_samples, axis=0)

    def build_fix_caps(self):
        heading_sample = self.sample_heading_time * self.sample_rate
        self.sample_heading = np.blackman(int(heading_sample * 2))[:int(heading_sample)]
        ending_sample = self.sample_fix_ending_time * self.sample_rate
        self.sample_fix_ending = np.blackman(int(ending_sample * 2))[-int(ending_sample):]

    def build_dynamic_end_caps(self, pitch):
        ending_sample = self.piano_max_ending_sample_time * self.piano_max_ending_sample_power_param ** (
                -pitch / 88)
        ending_sample *= self.sample_rate
        sample_ending = np.hanning(int(ending_sample * 2))[-int(ending_sample):]
        return sample_ending

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

    def get_start_idx(self):
        return int(random.uniform(0, self.random_sample_shift * self.sample_rate))

    def get_piano_volume_decay(self, key, velocity, length):
        x = np.array([i / self.sample_rate for i in range(length)])
        volume = (1 - self.piano_final_energy) * (
                2 + self.piano_volume_decay_param * key * velocity * (key / 88) ** 2) ** (-x) + self.piano_final_energy
        overtone_volume = (1 - self.piano_final_overtone) * (
                1 + self.piano_overtone_decay_param * key * velocity * (key / 88) ** 2) ** (
                              -x) + self.piano_final_overtone
        return volume, overtone_volume

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

    def get_sample_piano_single_mic(self, key, velocity, random_sample_shift_idx, sample_length, sample_ending,
                                    delta_time, left_mic=True):
        start_idx = self.calculate_start_idx(left_mic, delta_time)
        random_sample_shift_idx += start_idx
        sin_sample = self.sound_library_sin[key][
                     random_sample_shift_idx:sample_length + random_sample_shift_idx]
        high_velocity_sample = self.sound_library_high_velocity[key][
                               random_sample_shift_idx:sample_length + random_sample_shift_idx]
        # mix sound: volume low->high ==> sin->triangle/square
        decay_volume, overtone_decay_volume = self.get_piano_volume_decay(key, velocity, sample_length)
        final_decay_volume = overtone_decay_volume * np.power(velocity, self.piano_hard_sound_power_param)
        # trim sample
        valid_sample_length = min(len(final_decay_volume), len(sin_sample), len(high_velocity_sample))
        final_decay_volume = final_decay_volume[:valid_sample_length]
        sin_sample = sin_sample[:valid_sample_length]
        high_velocity_sample = high_velocity_sample[:valid_sample_length]
        decay_volume = decay_volume[:valid_sample_length]
        # mix sample
        sample = sin_sample * (1 - final_decay_volume) + high_velocity_sample * final_decay_volume
        # adding caps
        sample[:len(self.sample_heading)] *= self.sample_heading
        sample[-len(sample_ending):] *= sample_ending
        # apply velocity and volume decay
        sample = self.norm_audio(sample) * decay_volume * velocity
        return sample

    def get_sample_piano(self, key, velocity, note_duration):
        velocity /= 127
        # shift sample for phase variation
        random_sample_shift_idx = self.get_start_idx()
        sample_ending = self.build_dynamic_end_caps(key)
        sample_length = int(round(note_duration * self.sample_rate) + len(sample_ending)
                            - self.piano_max_ending_sample_pre_time)
        sample_length = max(sample_length, len(self.sample_heading) + len(sample_ending))
        delta_time, sound_power_ratio = self.spatial_modeling.get_parameter(key)
        left_sample = self.get_sample_piano_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                       sample_ending, delta_time, left_mic=True) * sound_power_ratio[0]
        right_sample = self.get_sample_piano_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                        sample_ending, delta_time, left_mic=True) * sound_power_ratio[1]
        sample = np.transpose([left_sample, right_sample])
        return sample

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
        delta_time, sound_power_ratio = self.spatial_modeling.get_parameter(key)
        left_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                 sample_ending, delta_time, left_mic=True) * sound_power_ratio[0]
        right_sample = self.get_sample_single_mic(key, velocity, random_sample_shift_idx, sample_length,
                                                  sample_ending, delta_time, left_mic=True) * sound_power_ratio[1]
        sample = np.transpose([left_sample, right_sample])
        return sample

    def create_empty_track(self, duration):
        return np.array([[0.0, 0.0] for _ in range(int(duration * self.sample_rate))])

    def synthesize(self, mid_path, out_path):
        # analyze midi
        mt = MidiTranslate(mid_path)
        duration = mt.music_duration
        if self.piano_sustain:
            mt.apply_sustain_pedal_to_notes()
        notes = mt.notes
        # create performance
        synthesized_audio = self.create_empty_track(duration + self.piano_max_ending_sample_time)
        for ch, channel_notes in notes.items():
            for n in channel_notes:
                if self.piano_sound:
                    sample = self.get_sample_piano(n.pitch, n.velocity, n.duration)
                else:
                    sample = self.get_sample(n.pitch, n.velocity, n.duration)
                starting_sample = int(round(n.start * self.sample_rate))
                synthesized_audio[starting_sample:starting_sample + len(sample)] += sample
        # apply reverb
        if self.reverb_add:
            synthesized_audio = self.rvb.apply(synthesized_audio, self.sample_rate)
        # norm audio
        synthesized_audio = self.norm_audio(synthesized_audio)
        # export
        sf.write(out_path, synthesized_audio, samplerate=self.sample_rate)


if __name__ == '__main__':
    import sys

    s = Synthesize()
    s.synthesize(sys.argv[1], sys.argv[2])
