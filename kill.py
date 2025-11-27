import socket

s = socket.socket()
s.connect(("127.0.0.1", 5001))
s.send(b"STOP")
s.close()

print("Stop signal sent.")
