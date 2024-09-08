import numpy as np
import zmq
import os
import pickle
import logging
import time
from tkinter import *
import pandas as pd
import copy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from datetime import datetime, timedelta
import matplotlib.dates as mdates

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
continuePlottingTotalsim=False

def get_numeric_cols(start_indx, end_indx, temp_shift):
    tmp = test_set[start_indx:end_indx, use_numeric_cols]
    tmp[:, 0] = tmp[:,0] + temp_shift   # add two degrees to the temp
    return tmp

def data_points(i):
    out = []
    for cat_col_indx in use_cat_cols:
        out.append(test_set[i:i+chunksize, cat_col_indx].astype('int32'))
    out.append(get_numeric_cols(i,i+chunksize,0))
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

def ts_int_to_datetime(x):
    x=(x/(24*3600))
    d = np.floor(x)
    day_seconds = (x-d) *(24*3600)
    return datetime(2000,1, 1 ) + timedelta(days=d, seconds=day_seconds)

def data_points_total(i):
    out = []
    for cat_col_indx in use_cat_cols:
        out.append(test_set[i:i+chunksize, cat_col_indx].astype('int32'))
    out.append(get_numeric_cols(i, i + chunksize,0))
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

def data_points_total_sim(i, temp_adj):
    out = []
    for cat_col_indx in use_cat_cols:
        out.append(test_set[i:i+chunksize, cat_col_indx].astype('int32'))
    out2 = copy.deepcopy(out)
    out.append(get_numeric_cols(i, i + chunksize,0))
    out2.append(get_numeric_cols(i, i + chunksize, temp_adj))
    if len(out)>0:
        out_s = pickle.dumps(out, protocol=0)
        logger.debug("sending message to server with row count " + str(len(out[0])))
        socket.send(out_s)
        logger.debug("waiting for response")
        message = socket.recv()
        message = pickle.loads(message)
        prediction = message[:,0]

        out_s = pickle.dumps(out2, protocol=0)
        logger.debug("sending message to server with row count " + str(len(out2[0])))
        socket.send(out_s)
        logger.debug("waiting for response")
        message = socket.recv()
        message = pickle.loads(message)
        prediction2 = message[:, 0]
        actual = test_set[i:i + chunksize, label_col]
        ts_int = test_set[i:i + chunksize, 9]
    return ts_int, prediction, prediction2, actual

def app():
    root = Tk()
    root.title("Western Power Hack-a-gig : London Smart Meter ")
    root.config(background='white')
    root.geometry("1000x700")
    bottom = Frame(root)
    bottom.pack(side=BOTTOM, fill=BOTH, expand=True)

    iconpth = os.path.join(os.path.dirname(__file__), "WesternPower.gif")
    icon = PhotoImage(file=iconpth)

    root.tk.call('wm','iconphoto',root._w,icon)

    fig = Figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel("Actual ")
    ax.set_ylabel("Predicted")
    ax.grid()

    graph = FigureCanvasTkAgg(fig, master=root)
    graph.get_tk_widget().pack(side="top", fill='both', expand=True)

    tkvar = StringVar(root)
    choices = {'-10', '-5', '-2', '2', '5','10'}
    tkvar.set('10')

    def change_dropdown(*args):
        logger.debug("changing in the temp adjuemnt to " + tkvar.get())
    tkvar.trace('w', change_dropdown)


    def plotter():
        start_row = 0
        while continuePlotting:
            ax.cla()
            ax.grid()
            predicted, actual = data_points(start_row)
            ax.plot(actual, predicted, 'ro', ms=1, color='green')
            ax.plot([0, 6], [0, 6], c="red", marker='.', linestyle=':')
            ax.set_title("Half Hour Meter Readings, Equivalence Plot ")
            ax.axis([0, 6, 0, 6])
            ax.set_xlabel("actual kwh")
            ax.set_ylabel("predicted kwh")
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
            ax.plot(df2['ts_int'], df2['prediction'], ms=1, linestyle=':', color='green', label="Predicted")
            ax.plot(df2['ts_int'], df2['actual'], ms=1,  color='red',  label="Actual")
            ax.legend(loc="upper right")
            ax.set_title("Aggregated All Meters")
            ax.set_xlabel("time (ts_int)")
            ax.set_ylabel("total kwh")
            graph.draw()
            time.sleep(1)
            start_row+=int(chunksize)

    def plotter_all_sim():
        start_row = 0
        while continuePlottingTotalsim:
            if start_row> len(test_set):
                start_row=0
            ax.cla()
            ax.grid()
            offset = copy.deepcopy(int(tkvar.get()))
            ts_int, predicted, predicted2,  actual = data_points_total_sim(start_row, offset)
            df = pd.DataFrame({'ts_int': ts_int, 'prediction': predicted, 'prediction2': predicted2, 'actual': actual})
            df2 = df.groupby(['ts_int'],as_index=False, sort=True).sum()
            df2['ts_labels']=df2['ts_int'].apply(lambda x:ts_int_to_datetime(x))

            ax.plot(df2['ts_labels'], df2['prediction'], ms=1, linestyle=':', color='green', label='Predicted')
            ax.plot(df2['ts_labels'], df2['prediction2'], ms=1,  linestyle='-.', color='green', label="Predicted(adj " + str(offset) + " degrees C)")
            ax.plot(df2['ts_labels'], df2['actual'], ms=1, color='red', label="Actual")
            ax.legend(loc="upper right")
            ax.set_title("All Meters")
            #ax.set_xticks(df2['ts_labels'])

            hours = mdates.HourLocator(byhour=[6,12,18])  # every year
            days = mdates.DayLocator()  # every month

            daysfmt = mdates.DateFormatter('%d/%m')
            hoursfmt = mdates.DateFormatter('%H:%M')

            # format the ticks
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_major_formatter(daysfmt)

            ax.xaxis.set_minor_locator(hours)
            ax.xaxis.set_minor_formatter(hoursfmt)

            ax.xaxis.set_tick_params(rotation=45)

            ax.set_xlabel("Time")
            ax.set_ylabel("total kwh")
            graph.draw()
            time.sleep(1)
            start_row+=int(chunksize)

    def gui_handler():
        global continuePlotting,continuePlottingTotal, continuePlottingTotalsim,chunksize
        chunksize = 1000
        continuePlottingTotal=False
        continuePlottingTotalsim=False
        continuePlotting = True
        threading.Thread(target=plotter).start()

    def gui_handler_Total():
        global continuePlotting, continuePlottingTotal,continuePlottingTotalsim,chunksize
        chunksize=300000
        continuePlotting=False
        continuePlottingTotalsim = False
        continuePlottingTotal=True
        threading.Thread(target=plotter_all).start()

    def gui_handler_Total_sim():
        global continuePlotting, continuePlottingTotal,continuePlottingTotalsim,chunksize
        chunksize=1000000
        continuePlotting=False
        continuePlottingTotal=False
        continuePlottingTotalsim = True
        threading.Thread(target=plotter_all_sim).start()

    b = Button(root, text="Real-Time Meter Predictions", command=gui_handler,  fg="black")
    b3 = Button(root, text="Real-Time Total", command=gui_handler_Total_sim, fg="black")
    b.pack(in_=bottom, side=LEFT, fill=Y)

    b3.pack(in_=bottom, side=LEFT, fill=Y)
    popupMenu = OptionMenu(root, tkvar, *choices)
    Label(root, text="Set Temp Offset ").pack(in_=bottom, side=LEFT, fill=Y)
    popupMenu.pack(in_=bottom, side=LEFT, fill=Y)


    root.mainloop()

if __name__ == '__main__':
    app()