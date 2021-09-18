import argparse

# https://github.com/RobertBoganKang/file_processing
from file_processing import FileProcessing
# https://github.com/RobertBoganKang/midi_translate
from syntheize import Synthesize


class BatchSynthesize(FileProcessing):
    """
    batch apply synthesize performance
    """

    def __init__(self, ops):
        super().__init__(ops)
        self.sample_rate = ops.sample_rate
        self.reverb = ops.reverb
        self.sound_type = ops.sound_type
        self.long_notes = ops.long_notes
        self.synth = Synthesize(prepare_sound=False)

        self.change_part_synth_param()

    def change_part_synth_param(self):
        self.synth.sample_rate = self.sample_rate
        self.synth.reverb_add = self.reverb
        self.synth.high_velocity_type = self.sound_type
        if self.long_notes:
            self.synth.piano_sound = False
        self.synth.prepare_sound_library()

    def do(self, in_path, out_path):
        if os.path.exists(out_path):
            return
        self.synth.synthesize(in_path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='batch apply synthesize performance')
    fp_group = parser.add_argument_group('file processing arguments')
    fp_group.add_argument('--input', '-i', type=str, help='the input folder/file, or a text file for paths',
                          default='in')
    fp_group.add_argument('--in_format', '-if', type=str, help='the input format', default='mid')
    fp_group.add_argument('--output', '-o', type=str, help='the output folder/file', default='out')
    fp_group.add_argument('--out_format', '-of', type=str, help='the output format', default='wav')
    fp_group.add_argument('--cpu_number', '-j', type=int, help='cpu number of processing', default=0)

    synth_group = parser.add_argument_group('synthesize arguments')
    synth_group.add_argument('--sample_rate', '-sr', type=int, help='sample rate to process', default=48000)
    synth_group.add_argument('--reverb', '-rvb', action='store_false', help='apply reverb')
    synth_group.add_argument('--long_notes', '-ln', action='store_true', help='use long note style')
    synth_group.add_argument('--sound_type', '-s', type=str, help='sound type `triangle` or `square`',
                             default='triangle')

    args = parser.parse_args()

    # do batch synthesize
    BatchSynthesize(args)()
