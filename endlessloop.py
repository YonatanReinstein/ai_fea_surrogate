import socket
from multiprocessing import Pool
from time import sleep
import os

STOP = False   

def listener():
    global STOP
    s = socket.socket()
    s.bind(("127.0.0.1", 5001))
    s.listen(1)
    conn, addr = s.accept()
    msg = conn.recv(16)
    if msg == b"STOP":
        STOP = True
    conn.close()
    s.close()

def worker(x):
    import tempfile
    import shutil
    os.makedirs("tmp", exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="irit_", dir="tmp", )
    sleep(5)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return x*x

if __name__ == "__main__":
    import threading
    threading.Thread(target=listener, daemon=True).start()

    pool = Pool(1)
    it = pool.imap(worker, range(10000), chunksize=1)

    try:
        for result in it:
            if STOP:
                print("Graceful stop requested.")

                # -------- KEY PART --------
                pool.close()  
                sleep(10)      # do not accept new tasks
                pool.terminate()    # kill worker process
                # -------------------------

                break

            print("Result:", result)

    finally:
        pool.join()     # now join NEVER hangs
        print("Done.")
