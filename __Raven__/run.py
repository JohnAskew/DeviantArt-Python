''' To-DO
1. Done - Make me non-modal
2. Chg CPU MEM green text until > 50 then red. 

'''
import os, sys

sys.path.insert(0, "C:\\Users\\john\\Desktop\\python\\jerk" )

from import_modules import *


flags = QtCore.Qt.WindowFlags(QtCore.Qt.FramelessWindowHint)

def pprint_ntuple(nt):
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2human(value)
        print('%-10s : %7s' % (name.capitalize(), value))


def mainSYS():
    print('MEMORY\n------')
    pprint_ntuple(psutil.virtual_memory())
    print('\nSWAP\n----')
    pprint_ntuple(psutil.swap_memory())


###################################
class mainT(QThread):
###################################
    #--------------------------------------
    def __init__(self):
    #--------------------------------------
        super(mainT,self).__init__()

    @pyqtSlot()
    def run(self):

        pass


FROM_MAIN,_ = loadUiType(os.path.join(os.path.dirname(__file__), "./dragon_short.ui"))

###################################
class Main(QMainWindow, FROM_MAIN):
###################################



    def __init__(self, parent=None):

        super(Main, self).__init__(parent)

        self.setupUi(self)

        #self.setFixedSize(1200,740)

        self.setFixedSize(919, 707)

        self.setWindowTitle("__Raven__ The Digital Assistant")

        self.label_7 = QLabel

        self.label_8 = QLabel

        self.label_9 = QLabel

        self.label_10 = QLabel

        #self.setWindowFlag(flags)
        self.setWindowFlags(
        QtCore.Qt.FramelessWindowHint |
        QtCore.Qt.Window |
        QtCore.Qt.CustomizeWindowHint |
        QtCore.Qt.WindowTitleHint |
        QtCore.Qt.WindowCloseButtonHint 
        #Askew20210117 QtCore.Qt.WindowStaysOnTopHint
        )

        self.label_7 = QMovie("./lib/plasma.gif", QByteArray(), self)

        self.label_7.setCacheMode(QMovie.CacheAll)

        self.label_4.setMovie(self.label_7)

        self.label_8 = QMovie("./lib/blue_circle_2.gif", QByteArray(), self)

        self.label_8.setCacheMode(QMovie.CacheAll)

        self.label_circle1.setMovie(self.label_8)

        self.label_9 = QMovie("./lib/blue_circle_2.gif", QByteArray(), self)

        self.label_9.setCacheMode(QMovie.CacheAll)



        self.label_circle2.setMovie(self.label_9)



        self.label_10 = QMovie("./lib/circle_purple.gif", QByteArray(), self)

        self.label_10.setCacheMode(QMovie.CacheAll)

        self.label_circle_purple.setMovie(self.label_10)


        self.label_11 = QMovie("./lib/periscope.gif", QByteArray(), self)

        #self.label_11 = QMovie("./lib/pulse_circle.gif", QByteArray(), self)

        self.label_11.setCacheMode(QMovie.CacheAll)

        self.label_gif_dashboard.setMovie(self.label_11)




        self.dt = time.strftime("%A, %B, %d")

        self.label.setPixmap(QPixmap("./lib/bg_new_1.png"))


        self.cpu_engine = cpuinfo.get_cpu_info()["brand"]
        
        self.label_box_cpu_engine.setText("<font size=5 color='white'>" + self.cpu_engine + "</font>")

        self.label_box_cpu_engine.setFont(QFont(QFont('Acens', 5)))

        self.label_box_cpu_engine.setAlignment(Qt.AlignCenter)



        self.cpu_actual = str(cpuinfo.get_cpu_info()["hz_actual"])

        self.cpu_arch   = str(cpuinfo.get_cpu_info()["arch"])
        
        self.cpu_bit    = str(cpuinfo.get_cpu_info()["bits"])

        self.label_box_cpu_arch_bits.setText("<font size=5 color='white'>" + self.cpu_arch + " size: " + self.cpu_bit + "</font>")

        self.label_box_cpu_arch_bits.setFont(QFont(QFont('Acens', 5)))

        self.label_box_cpu_arch_bits.setAlignment(Qt.AlignLeft)


        self.label_box_cpu_speed.setText("<font size=5 color='white'>" + "Actual CPU speed: " + self.cpu_actual+ "</font>")

        self.label_box_cpu_speed.setFont(QFont(QFont('Acens', 5)))

        self.label_box_cpu_speed.setAlignment(Qt.AlignRight)



        self.label_6.setText("<font size=5 color='white'>" + self.dt + "</font>")

        self.label_6.setFont(QFont(QFont('Acens', 7)))

        self.label_6.setAlignment(Qt.AlignCenter)

        self.label_6tm.setAlignment(Qt.AlignCenter)

        self.label_6tm.setStyleSheet("color: #9575CD; background-color:transparent; font-weight:bold" ) #font 320px; font-weight:bold")

        self.label_6tm.setFont(QFont(QFont('Acens', 20)))

        self.shadow_effect = QGraphicsDropShadowEffect()

        self.shadow_effect.setColor(QColor("#FFFFFF"))

        self.shadow_effect.setOffset(1, 1)

        self.label_6tm.setGraphicsEffect(self.shadow_effect)



        self.label_text_cpu.setAlignment(Qt.AlignCenter)

        self.label_text_cpu.setText("<font size=4 color='white'>CPU</font>")

        self.label_text_cpu.setStyleSheet("color: #FFFFFF; background-color:transparent; font-weight:bold" ) #font 320px; font-weight:bold")

        self.label_text_cpu.setFont(QFont(QFont('Acens', 12)))

        self.shadow_effect = QGraphicsDropShadowEffect()

        self.shadow_effect.setColor(QColor("#037213"))

        self.shadow_effect.setOffset(2.0, 2.0)

        self.label_text_cpu.setGraphicsEffect(self.shadow_effect)



        self.label_text_mem.setAlignment(Qt.AlignCenter)

        self.label_text_mem.setText("<font size=4 color='white'>MEM</font>")

        self.label_text_mem.setStyleSheet("color: #FFFFFF; background-color:transparent; font-weight:bold" ) #font 320px; font-weight:bold")

        self.label_text_mem.setFont(QFont(QFont('Acens', 12)))

        self.shadow_effect = QGraphicsDropShadowEffect()

        self.shadow_effect.setColor(QColor("#C71408"))

        self.shadow_effect.setOffset(2.0, 2.0)

        self.label_text_mem.setGraphicsEffect(self.shadow_effect)



        self.label_box_cpu.setFont(QFont(QFont('Acens', 12)))

        self.label_box_cpu.setAlignment(Qt.AlignCenter)

        self.label_box_cpu.setStyleSheet("color: #036511; background-color:transparent; font-weight:bold" ) #font 320px; font-weight:bold")




        self.label_box_mem.setFont(QFont(QFont('Acens', 12)))

        self.label_box_mem.setAlignment(Qt.AlignCenter)

        self.label_box_mem.setStyleSheet("color: #C71408; background-color:transparent; font-weight:bold" ) #font 320px; font-weight:bold")

       
        

        self.timer = QTimer(self)

        self.timer.start(1000)

        self.timer.timeout.connect(self.show_time)

        self.timer = QTimer(self)

        self.timer.start(30000)

        self.timer.timeout.connect(self.show_cpu_pct)

        self.label_7.start()

        self.label_8.start()

        self.label_9.start()

        self.label_10.start()

        self.label_11.start()

        self.show()

    def show_time(self):

        current_time = QTime.currentTime()

        text = current_time.toString('hh:mm:ss')

        self.label_6tm.setText(text)

    def show_cpu_pct(self):
     
        cpu_pct = psutil.cpu_percent(1)

        mem = psutil.virtual_memory()[2]

        cpu_pct = str(cpu_pct)

        mem = str(mem)

        if float(cpu_pct) > float(50.0):

            self.label_box_cpu.setStyleSheet("color: #FF0000; background-color:transparent; font-weight:bold" )

        else:

            self.label_box_cpu.setStyleSheet("color: #036511; background-color:transparent; font-weight:bold" )

        if float(mem) > float(50.0):

            self.label_box_mem.setStyleSheet("color: #FF0000; background-color:transparent; font-weight:bold" )

        else:

            self.label_box_mem.setStyleSheet("color: #036511; background-color:transparent; font-weight:bold" )


        self.label_box_cpu.setText(cpu_pct)

        self.label_box_mem.setText(mem)



app = QtWidgets.QApplication(sys.argv)

mainSYS()

main = Main()

exit(app.exec_())
























