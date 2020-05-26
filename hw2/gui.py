from tkinter import *
from recorder import start_recording
from recorder import stop_recording
from model import predict
import threading

mock_file_path = "./data/demo/01.wav"

def start_reccord_gui():
    start_recording()

def stop_record_gui():
    stop_recording(mock_file_path)
    # label2['text'] = 'Calculating, patience is the key of doing right things'
    label2['text'] = 'Calculating'
    threading.Thread(target=pred).start()


def pred():
    label2['text'] = predict()

if __name__ == "__main__":
    window = Tk()

    button = Button(window, text="Start recording", command=start_reccord_gui)
    button2 = Button(window, text="Stop recording - Predict", command=stop_record_gui)
    # button3 = Button(window, text="Start recording", command=predict)
    label = Label(window, font=("Arial", 10), text="Predict:")
    label2 = Label(window, font=("Arial", 10))

    label.grid(row=0,column=0)
    label2.grid(row=0,column=1)

    button.grid(row=1,column=0)
    button2.grid(row=1,column=1)
    # button3.grid(row=1,column=2)


    window.title('Predict')
    window.mainloop()