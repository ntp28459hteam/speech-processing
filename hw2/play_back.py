import pyaudio
import wave
import os
import threading

from recorder import p

SAMPLE_FORMAT = pyaudio.paInt16
FRAME_RATE = 44100
CHUNK = 1024
CHANNELS = 2

frames = []
# p = None
stream = None
wf = None

is_playing = False
lock = threading.Lock()

# https://stackoverflow.com/questions/6951046/how-to-play-an-audiofile-with-pyaudio

def start_playback_impl(filepath):
    global p
    global is_playing
    global stream
    global wf

    # print(filepath)

    lock.acquire()
    try:
        if is_playing == True:
            # lock.release()
            return
        is_playing = True
        wf = wave.open(filepath, 'rb')
    finally:
        lock.release()
    if p is None:
        p = pyaudio.PyAudio()
    if stream is None:
        stream = p.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=int(FRAME_RATE),
            frames_per_buffer=CHUNK,
            output=True)
    threading.Thread(target=play).start()
    print('Start playback')

def stop_playback_impl():
    global is_playing
    global p
    global frames
    global stream

    lock.acquire()
    try:
        if is_playing == False:
            # lock.release()
            return
        is_playing = False
    finally:
        lock.release()

    # frames.clear()
    stream.close()
    stream = None
    # p.terminate()

    print('Stop playback')
def play():
    global stream
    global frames
    frames = wf.readframes(CHUNK)
    while frames != b'':
        stream.write(frames)
        frames = wf.readframes(CHUNK)
        # print(frames)
        lock.acquire()
        try:
            if is_playing is False:
                break
        finally:
            lock.release()