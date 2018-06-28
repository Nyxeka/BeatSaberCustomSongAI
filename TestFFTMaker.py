import os, wave, pylab, io
import soundfile as sf
import librosa as lb 



file = "C:\\Users\\njhyl\\Code\\ML\\songMaker\\songData\\song\\00000025.ogg"

y, sr = lb.load(file)

tempo, beat_frames = lb.beat.beat_track(y=y, sr=sr)

beat_times = lb.frames_to_time(beat_frames, sr=sr)

lb.output.times_csv('beat_times.csv', beat_times)

print(len(beat_times))

lb.output.write_wav("C:\\Users\\njhyl\\Code\\ML\\songMaker\\songData\\song\\00000025.wav",y,sr)

def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('spectrogram.png')
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# graph_spectrogram('C:\\TestData\\test.wav')