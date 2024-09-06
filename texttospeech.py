import os
import time

prevobj = "hello"

while True:
    # Read detected object from file
    f = open("/home/pi/ObjectDetection/obj.txt", "r")
    objinfo = f.read()
    f.close()

    if len(objinfo):
        if objinfo != prevobj:
            command = 'flite -t "' + objinfo + ' DETECTED"'
            print(command)
            os.system(command)
            prevobj = objinfo

    time.sleep(0.5)
