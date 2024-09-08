import inference
import zmq
from multiprocessing import Process
import pickle
import logging
import time
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter.converter = time.gmtime
ch.setFormatter(formatter)
logger.addHandler(ch)

class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        message = "I'm fine how are you ?"
        self.wfile.write(bytes(message, "utf8"))
        return

def HealthNew():
    logger.debug('starting health check server')
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    logger.debug('Serving Forever')
    httpd.serve_forever()

def ServerNew(port="5557"):
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.REP)
    server.bind('tcp://*:' + port)
    M = inference.MeterPredictor(logger)
    while True:
        try:
            logger.debug("waiting for request")
            message_pkl = server.recv()
            logger.debug("unpickle message")
            message = pickle.loads(message_pkl)
            logger.debug("making prediction")
            response = M.predict(message)
            logger.debug("sending reply")
        except:
            response = sys.exc_info()
            logger.debug(response)
        finally:
            response_s = pickle.dumps(response, protocol=0)
            server.send(response_s)

if __name__ == "__main__":
    Process(target=HealthNew).start()
    Process(target=ServerNew).start()