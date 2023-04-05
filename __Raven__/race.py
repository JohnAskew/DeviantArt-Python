import os, sys

try:

    import threading

    from threading import Thread

    from threading import Lock

except:

    os.system('pip install threading')

    import threading

    from threading import Thread

    from threading import Lock

try:

    import time

except:

    os.system('pip install time')

    import time


#--------------------------------------
def increase( lock=None, using_lock=False):
#--------------------------------------

    if using_lock:

        lock.acquire()

    global database_value

    local_copy = database_value

    local_copy +=1 

    time.sleep(0.1)

    database_value = local_copy

    if using_lock:

        lock.release()

#######################################
if __name__ == "__main__":
#######################################

    lock = Lock()
    
    global database_value

    database_value = 0

    global using_lock

    #
    # Here is the switch to turn locking off and on
    #
    #----------------------------------
    using_lock = False
    #----------------------------------
    #
    #

    print('start value: ' + str(database_value))

    if using_lock:

        print("Calling increase with locks")

        thread1 = Thread(target=increase, args=(lock, using_lock))

        thread2 = Thread(target=increase, args=(lock, using_lock))

    else:

        thread1 = Thread(target=increase)

        thread2 = Thread(target=increase)

    print("thread1 calling function increase with database_value: " + str(database_value))

    thread1.start()

    print("thread2 calling function increase with database_value: " + str(database_value))

    thread2.start()

    thread1.join()

    thread2.join()

    print("*** Ending program ***\n\nDoes database_value end with 2, since we called increase function twice?")

    print("\n#--------------------------------------")

    print('main ending with database_value: --> ' + str(database_value))

    print("#--------------------------------------")


