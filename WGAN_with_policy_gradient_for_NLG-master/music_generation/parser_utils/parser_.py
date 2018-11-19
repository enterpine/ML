# import cPickle
import midi
import numpy as np
# from __future__ import print_function
import glob


#######################################################################################
lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound
statematrix_dim1 = 156
statematrix_dim0 = 30


def midiToNoteStateMatrix(midi_file_path, squash=True, span=span):
    pattern = midi.read_midifile(midi_file_path)

    time_left = []
    for track in pattern:
        time_left.append(track[0].tick)
        # print("track[0].tick", track[0].tick)


    posns = [0 for track in pattern]
    # print("posns", posns)

    statematrix = []
    time = 0

    state = [[0, 0] for x in range(span)]
    # print("span", span)
    # print("state", state)
    # print("pattern.resolution", pattern.resolution)

    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            # print("state", state)
            statematrix.append(state)
        for i in range(len(time_left)):
            if not condition:
                break
            while time_left[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lower_bound] = [0, 0]
                        else:
                            state[evt.pitch - lower_bound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        out = statematrix
                        condition = False
                        break
                try:
                    time_left[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    time_left[i] = None

            if time_left[i] is not None:
                time_left[i] -= 1

        if all(t is None for t in time_left):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    # statematrix = np.asarray(statematrix).tolist()
    return statematrix

#######################################################################################
def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upper_bound - lower_bound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + lower_bound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(filename), pattern)
#######################################################################################

#######################################################################################

# train_misi_list = glob.glob(r"../datasets/mixed_all_tain/*")
# # print(train_misi_list)
# print(len(train_misi_list))
#
# midi_statematrixs = []
# count = 0
# for f in train_misi_list:
#     count += 1
#     midi_statematrix = midiToNoteStateMatrix(f)
#     print(count, midi_statematrix.shape)
#     if midi_statematrix.shape[0] >= statematrix_dim0:
#         midi_statematrix = midi_statematrix[:statematrix_dim0] # to limit the length
#         midi_statematrixs.append(midi_statematrix)
#         print(count, midi_statematrix.shape)
#     else:
#         print("pass")
#
# np.save("midi_statematrixs.npy", midi_statematrixs)
# print("saved")
#######################################################################################
# midi_statematrixs = np.load("midi_statematrixs.npy")
# print(midi_statematrixs.shape)
#
# midi_statematrixs_flatten = []
# for i in range(midi_statematrixs.shape[0]):
#     midi_statematrix = midi_statematrixs[i]
#     midi_statematrix = midi_statematrix.reshape((-1, ))
#     midi_statematrix = midi_statematrix.tolist()
#     print(i, len(midi_statematrix))
#     midi_statematrixs_flatten.append(midi_statematrix)
#
# np.save("midi_statematrixs_flatten.npy", midi_statematrixs_flatten)
#######################################################################################
# midi_statematrixs_flatten = np.load("midi_statematrixs_flatten.npy")
# print(midi_statematrixs_flatten.shape)
# for i in range(midi_statematrixs_flatten.shape[0]):
#     tmp = midi_statematrixs_flatten[i].reshape(statematrix_dim0, statematrix_dim1)
#     noteStateMatrixToMidi(tmp, filename="../outputs/train_statematrix/statematrix_{}".format(i))
#######################################################################################
# midi_statematrixs_flatten = np.load("midi_statematrixs_flatten.npy")
# print(midi_statematrixs_flatten.shape)
# f = open("midi_statematrixs_flatten_merge.txt", "w")
#
#
# for i in range(midi_statematrixs_flatten.shape[0]):
#     tmp = midi_statematrixs_flatten[i].tolist()
#     # print(len(tmp))
#     tmp = "".join([str(t) for t in tmp])
#     f.write(tmp)
#     print(i, tmp)
# f.close()

#######################################################################################
# f = open("midi_statematrixs_flatten.txt", "r")
# statematrix  = f.readlines()
# print(len(statematrix))
# print(len(statematrix[2]))
# f.close()

#######################################################################################


