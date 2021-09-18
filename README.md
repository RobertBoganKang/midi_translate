# MIDI  Translate

## Introduction

This project will translate midi file (`.mid`) into `Note`  and `Control` object, and can be used in other purposes.

## Requirement

Python library `mido`, `matplotlib`, `soundfile`, `pyroomacoustics` will be required.

## Demo

### Translate

Translate midi file into another format (not serialized).

```bash
python translate.py <path.mid>
```

### Synthesize

Synthesize midi file into synthesized performance (`piano` or `long note` style).

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

Note parameters:

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
    <channel_0>: [Control_00, Control_01...],
    <channel_1>: [Control_10...],
    ...
}
```

Control parameters:

* `channel`, midi channel;
* `control`, control function number;
* `value`, value of control;
* `start`, `end`, start and end time of notes;
* `tick_start`, `tick_end`, start and end tick of notes;
* `duration`, duration of notes;
* `ticks`, number of ticks of notes;

#### Apply Sustain Pedal

```python
mt.apply_sustain_pedal_to_notes()
```

#### Piano Roll

Plot piano-roll, the piano roll and tempo will be shown.

```python
mt.pianoroll(<start_time>, <end_time>)
```

The parameter `start_time` and `end_time` can be none (not given).

### File `synthesize.py`

#### Initialize

Initialize class with:

```python
s = Synthesize()
```

The program will synthesize midi files (`.mid` to `.wav`):

```python
synthesize(<path.mid>, <out_path.wav>)
```

#### Parameters Change

Some parameters could be changed for advanced functions.

* `self.sample_rate`, change the audio sample rate;
* `self.high_velocity_type`, `triangle` or `square` for different sound;
* `self.piano_sound`, if `False`, it switch to `long note` style;
* `self.piano_sustain`, if `True`, sustain pedal will be added;
* `self.reverb_add`, if `True`, reverberate effect will be add;

### File `batch_synth.py`

Apply `synthesize.py` in batch way.

```bash
python3 batch_synth.py -i <in_folder> -o <out_folder>
```

