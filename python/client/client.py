import socket
import os
import sys
import struct
import Encoder

def socket_client(filename,q,port_num):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', port_num))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024))
    cc = Encoder.imageEncoder(filename,q)
    filepath = "TEST.bin"
    if os.path.isfile(filepath):

        fhead = struct.pack('128sl', os.path.basename(filepath).encode('utf-8'), os.stat(filepath).st_size)
        s.send(fhead)
        fp = open(filepath, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                print('{0} file send over...'.format(os.path.basename(filepath)))
                break
            s.send(data)
        s.close()
    return cc

if __name__ == '__main__':
    socket_client('./source_data/image1.512',5,8888)
