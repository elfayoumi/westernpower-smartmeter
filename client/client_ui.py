import numpy as np
import zmq
import os
import pickle
import logging
import time
from tkinter import *
import pandas as pd

# these two imports are important
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

test_set = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference_server/inference/weights/test_set.npy"))
use_cat_cols = [0, 21, 18, 20, 19]
use_numeric_cols = [12, 17, 16, 38, 33, 29, 30]
label_col = test_set.shape[1]-1

test_set = test_set[test_set[:,0]<5566] # drop this meter ...

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter.converter = time.gmtime
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.debug("--------------------STARTING CLIENT PROCESS-------------------------")
context = zmq.Context()
logger.debug("Connecting to server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://NETLB-2852e358413fd72e.elb.us-east-1.amazonaws.com:%s" % 5557)
chunksize = 1000
continuePlotting = False
continuePlottingTotal=False

def data_points(i):
    out = []
    for cat_col_indx in use_cat_cols:
        out.append(test_set[i:i+chunksize, cat_col_indx].astype('int32'))
    out.append(test_set[i:i+chunksize, use_numeric_cols])
    if len(out)>0:
        out_s = pickle.dumps(out, protocol=0)
        logger.debug("sending message to server with row count " + str(len(out[0])))
        socket.send(out_s)
        logger.debug("waiting for response")
        message = socket.recv()
        message = pickle.loads(message)
        prediction = message[:,0]
        actual = test_set[i:i+chunksize,label_col]
    return prediction, actual

def data_points_total(i):
    out = []
    for cat_col_indx in use_cat_cols:
        out.append(test_set[i:i+chunksize, cat_col_indx].astype('int32'))
    out.append(test_set[i:i+chunksize, use_numeric_cols])
    if len(out)>0:
        out_s = pickle.dumps(out, protocol=0)
        logger.debug("sending message to server with row count " + str(len(out[0])))
        socket.send(out_s)
        logger.debug("waiting for response")
        message = socket.recv()
        message = pickle.loads(message)
        prediction = message[:,0]
        actual = test_set[i:i+chunksize,label_col]
        ts_int = test_set[i:i + chunksize, 9]
    return ts_int, prediction, actual

def app():
    root = Tk()
    root.title("Western Power Hack-a-gig : London Smart Meter ")
    root.config(background='white')
    root.geometry("1000x700")
    bottom = Frame(root)
    bottom.pack(side=BOTTOM, fill=BOTH, expand=True)
    Label(root, text="Meter Predictions", bg='white').pack()

    fig = Figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel("Actual ")
    ax.set_ylabel("Predicted")
    ax.grid()

    graph = FigureCanvasTkAgg(fig, master=root)
    graph.get_tk_widget().pack(side="top", fill='both', expand=True)

    def plotter():
        start_row = 1
        while continuePlotting:
            ax.cla()
            ax.grid()
            predicted, actual = data_points(start_row)
            ax.plot(actual, predicted, 'ro', ms=1, color='green')
            ax.plot([0, 6], [0, 6], c="red", marker='.', linestyle=':')
            ax.axis([0, 6, 0, 6])
            ax.set_xlabel("actual")
            ax.set_ylabel("predicted")
            graph.draw()
            time.sleep(1)
            start_row+=int(chunksize/2)

    def plotter_all():
        start_row = 0
        while continuePlottingTotal:
            if start_row> len(test_set):
                start_row=0
            ax.cla()
            ax.grid()
            ts_int, predicted, actual = data_points_total(start_row)
            df = pd.DataFrame({'ts_int': ts_int, 'prediction': predicted, 'actual': actual})
            df2 = df.groupby(['ts_int'],as_index=False, sort=True).sum()
            ax.plot(df2['ts_int'], df2['prediction'], ms=1, linestyle=':', color='green')
            ax.plot(df2['ts_int'], df2['actual'], ms=1,  color='red')
            ax.set_xlabel("ts_int")
            ax.set_ylabel("total_kwh")
            graph.draw()
            time.sleep(1)
            start_row+=int(chunksize)

    def gui_handler():
        global continuePlotting,continuePlottingTotal, chunksize
        chunksize = 1000
        continuePlottingTotal=False
        continuePlotting = True
        threading.Thread(target=plotter).start()

    def gui_handler_Total():
        global continuePlotting, continuePlottingTotal,chunksize
        chunksize=300000
        continuePlotting=False
        continuePlottingTotal=True
        threading.Thread(target=plotter_all).start()

    b = Button(root, text="Real-Time Meter Predictions", command=gui_handler,  fg="black")
    b2 = Button(root, text="Real-Time Total", command=gui_handler_Total,  fg="black")
    b.pack(in_=bottom, side=LEFT, fill=Y)
    b2.pack(in_=bottom, side=LEFT, fill=Y)
    root.mainloop()

if __name__ == '__main__':
    app()