import os

import mido
import numpy as np


class Note(object):
    """ object of notes """

    def __init__(self, channel=0, pitch=49, velocity=0, start=0, end=0, tick_start=0, tick_end=0):
        self.channel = channel
        self.pitch = pitch
        self.velocity = velocity
        self.start = start
        self.end = end
        self.tick_start = tick_start
        self.tick_end = tick_end
        self.duration = 0
        self.ticks = 0

    def update(self):
        self.duration = self.end - self.start
        self.ticks = self.tick_end - self.tick_start


class Control(object):
    """ object of controls """

    def __init__(self, channel=0, control=64, value=0, start=0, end=0, tick_start=0, tick_end=0):
        self.channel = channel
        self.control = control
        self.value = value
        self.start = start
        self.end = end
        self.tick_start = tick_start
        self.tick_end = tick_end
        self.duration = 0
        self.ticks = 0

    def update(self):
        self.duration = self.end - self.start
        self.ticks = self.tick_end - self.tick_start


class Beat(object):
    def __init__(self, bar=0, beat_in_bar=0, numerator=4, denominator=4, tick=0, time=0):
        self.bar = bar
        self.beat_in_bar = beat_in_bar
        self.numerator = numerator
        self.denominator = denominator
        self.tick = tick
        self.time = time


class MidiTranslate(object):
    """
    midi translate:
        * convert midi file into objects for audio properties
        * plot piano-roll & tempo
    get more midi for demo:
        midi ref: https://github.com/RobertBoganKang/midi_files
    """

    def __init__(self, mid_path):
        self.mid_path = mid_path
        # global params
        self.mid = None
        self.tpb = 0
        self.music_duration = 0
        self.total_ticks = 0
        self.total_bars = -1

        # program params
        self.note_recipe = {}
        self.tempo_recipe = []
        self.control_recipe = {}
        self.time_signature_recipe = []
        self.delta_times = []
        self.tick_to_second = []

        # program result params
        self.notes = {}
        self.controls = {}
        self.beats = []

        # piano roll params
        self.pianoroll_min_alpha = 0.2
        self.piano_key_color = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        self.default_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.sustain_pedal_tolerance_duration = 0.15
        self.pianoroll_figure_size = (16, 4)

        # do translate
        self.initialize()
        self.translate()

    def initialize(self):
        self.mid = mido.MidiFile(self.mid_path, clip=True)
        self.tpb = self.mid.ticks_per_beat
        self.music_duration = self.mid.length
        self.time_signature_recipe = [4, 4, 0, self.tpb]

    def one_tick_to_delta_time(self, tempo_tpu):
        """
        convert tick to time in seconds
        :param tempo_tpu: int; tick per millisecond
        :return: float; time in seconds per tick
        """
        tpu_to_second = tempo_tpu / 1e6
        beat_per_tick = 1 / self.tpb
        return beat_per_tick * tpu_to_second

    def delta_time_to_bpm(self, delta):
        """
        convert delta time (second per tick) to beat per minute (bpm)
        :param delta: float; delta time in second
        :return: float; bpm
        """
        return 60 / (delta * self.tpb)

    def note_on_operation(self, msg, tick):
        channel = msg.channel
        note = msg.note
        velocity = msg.velocity
        # note_on as note_off operation if velocity is 0
        if velocity == 0:
            self.note_off_operation(msg, tick)
            return
        # if channel not exist, get channel
        if channel not in self.note_recipe:
            self.note_recipe[channel] = {}
        # queue implementation
        if note not in self.note_recipe[channel]:
            self.note_recipe[channel][note] = [(tick, velocity)]
        else:
            self.note_recipe[channel][note].insert(0, (tick, velocity))

    def note_off_operation(self, msg, tick):
        channel = msg.channel
        note = msg.note
        next_tick = tick
        if note not in self.note_recipe[channel]:
            raise AttributeError('ERROR: midi format error for notes!')
        starting_tick, velocity = self.note_recipe[channel][note].pop()
        ending_tick = next_tick
        new_note = Note(channel=channel, pitch=note - 20, velocity=velocity, tick_start=starting_tick,
                        tick_end=ending_tick)
        if channel not in self.notes:
            self.notes[channel] = [new_note]
        else:
            self.notes[channel].append(new_note)

    def set_tempo_operation(self, msg, tick):
        tempo = msg.tempo
        self.tempo_recipe.append((tick, self.one_tick_to_delta_time(tempo)))

    def control_change_operation(self, msg, tick):
        channel = msg.channel
        control = msg.control
        value = msg.value
        # if channel not exist, get channel
        if channel not in self.control_recipe:
            self.control_recipe[channel] = {}
        # get controls
        if control not in self.control_recipe[channel]:
            last_channel = channel
            last_tick = 0
            last_value = 0
        else:
            last_channel = self.control_recipe[channel][control][0]
            last_tick = self.control_recipe[channel][control][1]
            last_value = self.control_recipe[channel][control][2]
        self.control_recipe[channel][control] = (channel, tick, value)
        control_obj = Control(channel=channel, control=control, value=last_value, tick_start=last_tick, tick_end=tick)
        if control_obj.tick_start < control_obj.tick_end and last_channel == channel:
            if channel not in self.controls:
                self.controls[channel] = {}
            if control not in self.controls[channel]:
                self.controls[channel][control] = []
            self.controls[channel][control].append(control_obj)

    def add_last_control(self):
        """ add controls for the last """
        for channel in self.control_recipe.keys():
            for control, values in self.control_recipe[channel].items():
                _, tick_start, value = values
                control_obj = Control(channel=channel, control=control, value=value, tick_start=tick_start,
                                      tick_end=self.total_ticks)
                if control_obj.tick_start < control_obj.tick_end:
                    if channel not in self.controls:
                        self.controls[channel] = {}
                    if control not in self.controls[channel]:
                        self.controls[channel][control] = []
                    self.controls[channel][control].append(control_obj)

    def time_signature_operation(self, msg, tick):
        last_numerator, last_denominator, last_tick, last_tpb = self.time_signature_recipe
        i = 0
        for t in range(last_tick, tick, last_tpb):
            beat_in_bar = i % last_numerator
            if beat_in_bar == 0:
                self.total_bars += 1
            self.beats.append(
                Beat(bar=self.total_bars, beat_in_bar=beat_in_bar,
                     numerator=last_numerator, denominator=last_denominator, tick=t))
            # update
            i += 1
        if msg is not None:
            # put current to recipe
            numerator = msg.numerator
            denominator = msg.denominator
            notated_32nd_notes_per_beat = msg.notated_32nd_notes_per_beat
            mid_beat_per_bar = int(32 / notated_32nd_notes_per_beat)
            music_tpb = int(mid_beat_per_bar / denominator * self.tpb)
            self.time_signature_recipe = [numerator, denominator, tick, music_tpb]

    def create_tempo_tick_to_second(self):
        tempo_delta_times = []
        tick = 0
        for i in range(len(self.tempo_recipe)):
            if i < len(self.tempo_recipe) - 1:
                next_tick = self.tempo_recipe[i + 1][0]
            else:
                next_tick = self.total_ticks
            while tick < next_tick:
                tempo_delta_times.append(self.tempo_recipe[i][1])
                tick += 1
        self.delta_times = tempo_delta_times
        self.tick_to_second = np.add.accumulate([0] + tempo_delta_times)

    def update_notes_time(self):
        note_obj = self.notes
        for ch in note_obj.keys():
            for i in range(len(note_obj[ch])):
                obj = note_obj[ch][i]
                tick_start = obj.tick_start
                tick_end = obj.tick_end
                start = self.tick_to_second[tick_start]
                end = self.tick_to_second[tick_end]
                obj.start = start
                obj.end = end
                obj.update()
            note_obj[ch].sort(key=lambda x: x.start)

    def update_controls_time(self):
        control_obj = self.controls
        for ch in control_obj.keys():
            for ctrl in control_obj[ch].keys():
                for i in range(len(control_obj[ch][ctrl])):
                    obj = control_obj[ch][ctrl][i]
                    tick_start = obj.tick_start
                    tick_end = obj.tick_end
                    start = self.tick_to_second[tick_start]
                    end = self.tick_to_second[tick_end]
                    obj.start = start
                    obj.end = end
                    obj.update()
                control_obj[ch][ctrl].sort(key=lambda x: x.start)

    def update_beat_time_and_bar(self):
        for beat in self.beats:
            tick = beat.tick
            beat.time = self.tick_to_second[tick]
        self.total_bars += 1

    @staticmethod
    def combine_sustain(sustain_ctrl):
        """
        ref: https://www.cnblogs.com/xiximayou/p/12346966.html
        """
        intervals = [[x.start, x.end] for x in sustain_ctrl if x.value > 0]
        res = []
        intervals.sort()
        for i in intervals:
            if not res or res[-1][1] < i[0]:
                res.append(i)
            else:
                res[-1][1] = max(res[-1][1], i[1])
        return res

    def apply_sustain_pedal_to_notes(self):
        """ apply sustain pedal to the notes """
        for ch, ch_notes in self.notes.items():
            if 64 not in self.controls[ch]:
                break
            ch_control = self.combine_sustain(self.controls[ch][64])
            for i in range(len(ch_notes)):
                note = self.notes[ch][i]
                note_end = note.end
                for ctrl in ch_control:
                    ctrl_start = ctrl[0]
                    ctrl_end = ctrl[1]
                    if ctrl_start - self.sustain_pedal_tolerance_duration < note_end < ctrl_end:
                        self.notes[ch][i].end = max(note_end, ctrl_end)
                        self.notes[ch][i].update()
                        break

    def translate(self):
        for i in range(len(self.mid.tracks)):
            track = self.mid.tracks[i]
            current_tick = 0
            for msg in track:
                string_msg = str(msg)
                event_time = msg.time
                next_tick = current_tick + event_time
                # notes cases
                if 'note_on' in string_msg:
                    self.note_on_operation(msg, next_tick)
                elif 'note_off' in string_msg:
                    self.note_off_operation(msg, next_tick)
                elif 'set_tempo' in string_msg:
                    self.set_tempo_operation(msg, next_tick)
                elif 'control_change' in string_msg:
                    self.control_change_operation(msg, next_tick)
                elif 'time_signature' in string_msg:
                    self.time_signature_operation(msg, next_tick)

                # update tick
                current_tick = next_tick
            self.total_ticks = max(current_tick, self.total_ticks)
        # add last status
        self.add_last_control()
        self.time_signature_operation(None, self.total_ticks)
        # build tempo index: tick -> time
        self.create_tempo_tick_to_second()
        self.update_notes_time()
        self.update_controls_time()
        self.update_beat_time_and_bar()

    def plot_pram_fix(self, start_time, end_time, export_folder):
        # check
        if not self.notes:
            print('WARNINT: empty track, cannot plot!')
            return
        # 0. fix x-axis range
        if start_time is None:
            start_time = 0
        else:
            start_time = max(start_time, 0)
        if end_time is None:
            end_time = self.music_duration
        else:
            end_time = min(end_time, self.music_duration)
        assert start_time < end_time
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
        return start_time, end_time, export_folder

    def plot_pianoroll(self, start_time=None, end_time=None, export_folder=None):
        """ plot notes """
        import matplotlib.pyplot as plt
        self.plot_pram_fix(start_time, end_time, export_folder)
        fig = plt.figure(figsize=self.pianoroll_figure_size)
        min_pitch = 1e3
        max_pitch = -1e3
        for ch in self.notes.keys():
            # plot sustain pedal region
            if 64 not in self.controls[ch]:
                break
            for i in range(len(self.controls[ch][64])):
                obj = self.controls[ch][64][i]
                if obj.value != 0 and obj.control == 64:
                    start = obj.start
                    end = obj.end
                    # sustain pedal region
                    plt.fill([start, end, end, start], [0, 0, 89, 89], facecolor='k', alpha=0.1,
                             zorder=2)
            # plot notes lines
            color = self.default_color[ch % len(self.default_color)]
            for key in range(len(self.notes[ch])):
                obj = self.notes[ch][key]
                pitch = obj.pitch
                velocity = obj.velocity / 127
                min_pitch = min(pitch, min_pitch)
                max_pitch = max(pitch, max_pitch)
                start = obj.start
                end = obj.end
                # notes
                plt.fill([start, end, end, start], [pitch - 0.5, pitch - 0.5, pitch + 0.5, pitch + 0.5],
                         facecolor=color, edgecolor=color,
                         alpha=velocity * (1 - self.pianoroll_min_alpha) + self.pianoroll_min_alpha,
                         zorder=2)
                # notes velocity line
                pitch_line_y = pitch - 0.5 + velocity
                plt.plot([start, end], [pitch_line_y, pitch_line_y], c=color, zorder=3)

        # plot grid-line
        for key in range(min_pitch - 1, max_pitch + 2):
            # black key
            if self.piano_key_color[key % 12 - 4] == 1:
                plt.fill([0, self.music_duration, self.music_duration, 0], [key - 0.5, key - 0.5, key + 0.5, key + 0.5],
                         alpha=0.3, facecolor='lightgray', zorder=0)
            # grid line
            plt.plot([0, self.music_duration], [key - 0.5, key - 0.5], c='lightgray', linewidth=0.5, zorder=1)
        # plot bar and beats
        for beat in self.beats:
            if beat.beat_in_bar == 0:
                alpha = 0.6
            else:
                alpha = 0.2
            plt.plot([beat.time, beat.time], [min_pitch - 1, max_pitch + 1], c='gray', linewidth=1, zorder=1,
                     alpha=alpha)
        # start end edge
        plt.plot([0, 0], [min_pitch - 1, max_pitch + 1], c='gray', linewidth=3, zorder=3)
        plt.plot([self.music_duration, self.music_duration], [min_pitch - 1, max_pitch + 1], c='gray', linewidth=3)
        # plot rest
        plt.xlabel('time (second)')
        plt.ylabel('pitch')
        plt.xlim(start_time, end_time)
        plt.ylim(min_pitch - 1, max_pitch + 1)
        if export_folder is not None:
            fig.savefig(os.path.join(export_folder, 'piano_roll.svg'), bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_tempo(self, start_time=None, end_time=None, export_folder=None):
        """ plot tempo """
        import matplotlib.pyplot as plt
        self.plot_pram_fix(start_time, end_time, export_folder)
        fig = plt.figure(figsize=self.pianoroll_figure_size)
        ss = [self.delta_time_to_bpm(x) for x in self.delta_times]
        xx = []
        for key in range(len(self.delta_times)):
            xx.append(self.tick_to_second[key])
        # plot bar and beats
        beat_range = max(10, max(ss) - min(ss))
        for beat in self.beats:
            if beat.beat_in_bar == 0:
                alpha = 0.6
            else:
                alpha = 0.2
            plt.plot([beat.time, beat.time], [min(ss) - 0.05 * beat_range, max(ss) + 0.05 * beat_range],
                     c='gray', linewidth=1, zorder=1, alpha=alpha)
        plt.plot(xx, ss, c='k')
        plt.xlabel('time (second)')
        plt.ylabel('tempo (bpm)')
        plt.xlim(start_time, end_time)
        plt.ylim(min(ss) - 0.05 * beat_range, max(ss) + 0.05 * beat_range)
        if export_folder is not None:
            fig.savefig(os.path.join(export_folder, 'tempo.png'), bbox_inches='tight')
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    import sys

    mt = MidiTranslate(sys.argv[1])
    # test translate
    # notes
    for note_ch, notes in mt.notes.items():
        print(f'--CHANNEL[{note_ch}]--')
        for n in notes:
            print(f'PITCH:{str(n.pitch).rjust(3)}\t'
                  f'VELOCITY:{str(n.velocity).rjust(3)}\t'
                  f'START:{round(n.start, 3)}\t'
                  f'END:{round(n.end, 3)}\t'
                  f'DURATION:{round(n.duration, 3)}')
    # controls
    print('-' * 80)
    for control_ch, controls in mt.controls.items():
        print(f'--CHANNEL[{control_ch}]--')
        for control_num in controls.keys():
            print(f'<{control_num}>')
            for ii in range(len(controls[control_num])):
                c = controls[control_num][ii]
                print(f'CONTROL:{str(c.control).rjust(3)}\t'
                      f'VALUE:{str(c.value).rjust(3)}\t'
                      f'START:{round(c.start, 3)}\t'
                      f'END:{round(c.end, 3)}\t'
                      f'DURATION:{round(c.duration, 3)}')
    # beats
    print('-' * 80)
    for b in mt.beats:
        print(f'BAR:{b.bar}\t'
              f'BEAT_IN_BAR:{b.beat_in_bar}\t'
              f'TEMPO:{b.numerator}/{b.denominator}\t'
              f'TICK:{b.tick}\t'
              f'TIME:{round(b.time, 3)}')

    # plot piano-roll
    mt.plot_pianoroll()
    mt.plot_tempo()
