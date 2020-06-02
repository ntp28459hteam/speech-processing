from tkinter import *
from recorder import start_recording
from recorder import stop_recording
from play_back import start_playback_impl
from play_back import stop_playback_impl
from model import predict
import threading
import sounddevice as sd

mock_file_path = "./data/demo/01.wav"

def start_reccord_gui():
    start_recording()

def stop_record_gui():
    stop_recording(mock_file_path)
    # label2['text'] = 'Calculating, patience is the key of doing right things'
    label2['text'] = 'Calculating'
    threading.Thread(target=pred_gui).start()

# def start_playback_gui():
#     start_playback_impl(mock_file_path)
# def stop_playback_gui():
#     stop_playback_impl()

def pred_gui():
    if checkCmd.get() == 0:
        label2['text'] = predict(load_from_disk=False)
    else:
        label2['text'] = predict(load_from_disk=True)

if __name__ == "__main__":

    # misc = sd.query_devices()
    # print(misc)
    # sd.default.device = 10
    # print (misc[10])

    window = Tk()

    button = Button(window, text="Start recording", command=start_reccord_gui)
    label = Label(window, font=("Arial", 10), text="Predict:")
    label2 = Label(window, font=("Arial", 10))

    label.grid(row=0,column=0)
    label2.grid(row=0,column=1)

    button.grid(row=1,column=0)



    # button3 = Button(window, text="Start playback", command=start_playback_gui)
    # button4 = Button(window, text="Stop playback", command=stop_playback_gui)
    # button3.grid(row=2,column=0)
    # button4.grid(row=2,column=1)


    checkCmd = IntVar()
    checkCmd.set(1)

    labelText = Label(window)
    checkBox1 = Checkbutton(window, text="Load model from disk",variable=checkCmd, onvalue=1, offvalue=0)
    button2 = Button(window, text="Predict", command=stop_record_gui)

    labelText.grid(row=2,column=0)
    checkBox1.grid(row=3,column=0)
    button2.grid(row=3,column=1)

    window.title('Predict')
    window.mainloop()