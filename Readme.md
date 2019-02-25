In: 1024 floats (beat times, pad with 0's on either end)

Out: 1024 uint8, 1024 floats (192 possibilities per note in uint8, ignore anything past 192, float represents the time that the note sits at).

common terms:

xydt: note x position, y position, direction and time.

beat-time: Using a python library to map out beat times for a sound file.