import os, sys

try:

    import threading as th

except:

    os.system("pip install threading")

    import threading as th

try:

    from six.moves import input

except:

    os.system('pip install six.moves')

    from six.moves import input

try:

    import signal

except:

    os.system('pip install signal')

    import signal

app = "google"

lock = f'{app}_lock'

#--------------------------------------
def signal_handler(signum, frame):
#--------------------------------------
    print('signal_handler called with signal', signum)

    exit_event.set()

#--------------------------------------
def do_this():
#--------------------------------------

    global x

    x = 0

    print("This is our thread: " + str(app_thread) + "\n")

    while (not dead):

        if exit_event.is_set():

            print("do_this is breaking.")

            break

        x +=1

        if x > 299999:

            signal_handler(signal.SIGTERM)



#--------------------------------------
def main():
#--------------------------------------
    
    global x

    x = 0

    global dead

    dead = False

    global app_thread

    app_thread =  f'{app}_thread'

    print("main code, app_tread = " + str(app_thread))

    global exit_event

    exit_event = th.Event()

    signal.signal(signal.SIGTERM, signal_handler)

    our_thread = th.Thread(target=do_this, name=f'{app}_thread',)

    our_thread.start()

    print(th.enumerate())

    print(th.active_count())

    print(f'{app}_thread is alive: ' + str(our_thread.is_alive()))

    input("Press Enter to die...")

    dead = True

    print("x = " + str(x))

    sys.exit(0)

if __name__ == "__main__":

    main()

