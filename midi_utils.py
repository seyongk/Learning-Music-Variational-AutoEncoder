# https://github.com/arman-aminian/lofi-generator
import glob
import pickle

import numpy as np
import pandas as pd
from music21 import chord, converter, instrument, note, pitch, stream

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.


def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append(
                [
                    np.round(element.offset / 0.25),
                    np.round(element.quarterLength / 0.25),
                    element.pitch.midi,
                ]
            )
        elif isinstance(element, chord.Chord):
            stream_list.append(
                [
                    np.round(element.offset / 0.25),
                    np.round(element.quarterLength / 0.25),
                    element.sortAscending().pitches[-1].midi,
                ]
            )
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame(
        {
            "pos": np_stream_list.T[0],
            "dur": np_stream_list.T[1],
            "pitch": np_stream_list.T[2],
        }
    )
    df = df.sort_values(
        ["pos", "pitch"], ascending=[True, False]
    )  # sort the dataframe properly
    df = df.drop_duplicates(subset=["pos"])  # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length + 1, dtype=np.int16) + np.int16(
        MELODY_NO_EVENT
    )  # set array full of no events by default.
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            n = df[df.pos == i].iloc[0]  # pick the highest pitch at each semiquaver
            output[i] = n.pitch  # set note on
            output[i + n.dur] = MELODY_NOTE_OFF
    return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df["offset"] = df.index
    df["duration"] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = (
        df.duration.diff(-1) * -1 * 0.25
    )  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[["code", "duration"]]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = (
                note.Rest()
            )  # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream


def get_notes(path):
    """Get all the notes and chords from the midi files in the 'path' directory"""
    notes = []

    print(len(glob.glob(path + "/*.mid")))
    for file in glob.glob(path + "/*.mid"):
        midi = converter.parse(file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    with open("data/notes", "wb") as filepath:
        pickle.dump(notes, filepath)

    return notes


def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)


def onehot_initialization(a, ncols):
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


def prepare_sequences(notes, pitchnames, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    # map between notes and integers and back
    note_to_idx = dict((note, idx) for idx, note in enumerate(pitchnames))
    idx_to_note = dict((idx, note) for idx, note in enumerate(pitchnames))

    sequence_length = 32
    network_input = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        network_input.append([note_to_idx[char] for char in sequence_in])
    ncols = max(max(network_input)) + 1
    ncols = ncols + (ncols % 4)
    network_input = onehot_initialization(np.array(network_input), ncols)
    return network_input, idx_to_note


def create_midi(prediction_output, output_path, offset_step):
    """convert the output from the prediction to notes and create a midi file
    from the notes"""
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        #         offset += 0.5
        #         offset += 1.75
        offset += offset_step

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp=output_path)
