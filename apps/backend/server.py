#!/usr/bin/python
''' Python server to receive data from one or more HandActivities Server '''
import protocol
import numpy as np
import string
import gzip, zlib
import model
from socketserver import StreamRequestHandler
from socketserver import TCPServer

# Global Dataset Variables
DATASERVER_PORT = 55155

class HandActivitiesDataHandler(StreamRequestHandler):
    def readall(self, sz):
        out = []
        recvlen = 0
        while recvlen < sz:
            chunk = self.request.recv(sz - recvlen)
            if len(chunk) == 0:
                raise EOFError()
            recvlen += len(chunk)
            out.append(chunk)
        return b"".join(out)
        
    # Request Handler
    def handle_one(self):
        try:
            data_len = self.readall(protocol.int_len)
            np_len = np.frombuffer(data_len,dtype=protocol.int_dtype)
            data_len = int(np_len.tolist()[0])
            
            self.data = self.readall(data_len)
            
            # Decompress
            np_data = np.frombuffer(self.data,dtype='>f4')
    
            # Perform prediction
            X = np_data.reshape(1,-1)

            softmax_output = model.predict(X).astype('>f4')
            print(softmax_output)
            self.request.sendto(softmax_output.tobytes(),self.client_address)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    def handle(self):
        print("Got client: {0}".format(self.client_address))
        while 1:
            try:
                self.state = protocol.init
                self.handle_one()
            except Exception as e:
                import traceback
                traceback.print_exc()
                break

def run():
    HOST, PORT = '', DATASERVER_PORT
    TCPServer.allow_reuse_address = True
    server = TCPServer((HOST, PORT), HandActivitiesDataHandler)
    print ("Server started! Listening to port %d" % PORT)
    server.serve_forever()

run()