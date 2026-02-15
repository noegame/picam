# client.py
import socket
import struct
import numpy as np

SOCKET_PATH = "./tmp/python_c_socket"

# Création données
width = 64
height = 64

image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
int_list = np.array([1, 2, 3, 4, 5], dtype=np.int32)
float_list = np.array([1.1, 2.2, 3.3], dtype=np.float32)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(SOCKET_PATH)

# Envoi header
header = struct.pack("iiii", width, height, len(int_list), len(float_list))
sock.sendall(header)

# Envoi données
sock.sendall(image.tobytes())
sock.sendall(int_list.tobytes())
sock.sendall(float_list.tobytes())

sock.close()

"""
En attente de connexion...
Image: 64x64
Nb int: 5
Nb float: 3
Premier pixel: 167
Premier int: 1
Premier float: 1.100000

"""