# MIDI  Translate

## Introduction

This project will translate midi file (`.mid`) into `Note` , `Control`  and `Beat` object, and can be used in other purposes.

## Requirement

Python library `mido`, `matplotlib`, `soundfile`, `pyroomacoustics`, `scipy`(optional) will be required.

## Demo

### Translate

Translate midi file into another format (not serialized).

```bash
python translate.py <path.mid>
```

### Synthesize

Synthesize midi file into synthesized performance.

```bash
python synthesize.py <path.mid> <out_path.wav>
```

## Usage

### File `translate.py`

#### Initialize

Initialize class with:

```python
mt = MidiTranslate(<path.mid>)
```

#### Translated Result

Notes object extracted from `mt.notes` with format:

```python
{
    <channel_0>: [Note_00, Note_01...],
    <channel_1>: [Note_10...],
    ...
}
```

`Note` parameters:

* `channel`, midi channel;
* `pitch`, piano key number format;
* `velocity`, strength of notes to play;
* `start`, `end`, start and end time of notes;
* `tick_start`, `tick_end`, start and end tick of notes;
* `duration`, duration of notes;
* `ticks`, number of ticks of notes;

Controls object extracted from `mt.controls` with format:

```python
{
    <channel_0>: {
        <control_number_0>: {Control_00, ...},
        <control_number_1>: {Control_10, ...},
        ...
    },
    <channel_1>: {...}
    ...
}
```

`Control` parameters:

* `channel`, midi channel;
* `control`, control function number;
* `value`, value of control;
* `start`, `end`, start and end time of notes;
* `tick_start`, `tick_end`, start and end tick of notes;
* `duration`, duration of notes;
* `ticks`, number of ticks of notes;

`Beat` parameters:

* `bar`, bar number of music;
* `beat_in_bar`, the index of beat in the bar;
* `numberator`, `denominator`, defines the tempo of music;
* `tick`, the tick represent relative time;
* `time`, time that beat starts;

```python
[
    Beat_0, Beat_1, ...
]
```

#### Apply Sustain Pedal

```python
mt.apply_sustain_pedal_to_notes()
```

#### Piano Roll

Plot piano-roll, the piano roll and tempo will be shown.

```python
mt.pianoroll(<start_time>, <end_time>, <export_folder>)
```

The parameter `start_time` and `end_time` can be none (not given).

If `export_folder` not given, the program will plot directly.

### File `synthesize.py`

#### Initialize

Initialize class with:

```python
syn = Synthesize()
```

The program will synthesize midi files (`.mid` to `.wav`):

```python
syn.synthesize(<path.mid>, <out_path.wav>)
```

#### Parameters Change

Some parameters could be changed for advanced functions.

* `self.sample_rate`, change the audio sample rate (in `sound_library/utils.py::SoundCommon`);
* `self.piano_sustain`, if `True`, sustain pedal will be added;
* `self.reverb_add`, if `True`, reverberate effect will be add;
* `self.beat_sound_mix`, if `>0`, synthesize will mix beat samples;

#### Sound Library Change

```python
from sound_library.<your_sound_library> import SoundLibrary
```

Choose one sound library `<your_sound_library>`, then the sound will change.

There are several sound could choose:

* long notes;
* piano style 0;
* piano style 1;

### File `batch_synth.py`

Apply `synthesize.py` in batch way.

```bash
python3 batch_synth.py -i <in_folder> -o <out_folder>
```

