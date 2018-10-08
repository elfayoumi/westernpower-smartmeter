import numpy as np
import zmq
import os
import pickle
import logging
import time

test_set = np.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference_server/inference/weights/test_set.npy"))
use_cat_cols = [0, 21, 18, 20, 19]
use_numeric_cols = [12, 17, 16, 38, 33, 29, 30]

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
socket.connect ("tcp://NETLB-2852e358413fd72e.elb.us-east-1.amazonaws.com:%s" % 5557)
chunksize = 10000

for i in range (0,len(test_set), chunksize):
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
        logger.debug("response received with count " + str(len(message)))

logger.debug("----------------------CLOSING CLIENT PROCESS-------------------------")
socket.close()
exit(0)