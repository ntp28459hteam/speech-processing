import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import threading
import wave
import os
import sounddevice

# from keyboard_ import on_press
# from keyboard_ import listener
# from keyboard_ import is_recording

is_running = False
lock = threading.Lock()

SAMPLE_FORMAT = pyaudio.paInt16
FRAME_RATE = 44100
CHUNK = 1024
CHANNELS = 2

frames = []
p = None
stream = None


def stop_recording(file_name):
    global is_running
    global p
    global frames
    lock.acquire()
    try:
        if is_running == False:
            # lock.release()
            return
        is_running = False
    finally:
        lock.release()
    filename = file_name
    data = b''.join(frames)

    # #use wave to save data
    wf = wave.open(os.path.join("",filename), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(FRAME_RATE)
    wf.writeframes(data)
    wf.close()
    frames.clear()

    # global stream
    # stream.close()
    # stream = None
    # p.terminate()
    # p = None

    print('Stop recording')

def start_recording():
    global p
    global is_running
    global stream

    lock.acquire()
    try:
        if is_running == True:
            # lock.release()
            return
        is_running = True
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
            input=True)
    threading.Thread(target=record).start()
    print('Start recording')

def record():
    global stream
    while True:
        lock.acquire()
        try:
            if not is_running:
                stream.close()
                stream = None
                # lock.release()
                break
        finally:
            lock.release()
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
