import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from skimage import io, feature, color, transform
from compute_svm import main

class Ui_MainWindow(object):
    def __init__(self):       
        super(Ui_MainWindow, self).__init__()
        
    def setupUi(self, MainWindow):
        # Mainwindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 635)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # pushButton_openImage
        self.pushButton_openImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_openImage.setGeometry(QtCore.QRect(815, 10, 430, 50))
        self.pushButton_openImage.setObjectName("pushButton_openImage")
        #Radio button for points
        self.label_radiobheight = QtWidgets.QLabel(self.centralwidget)
        self.label_radiobheight.setGeometry(QtCore.QRect(820, 70, 430, 50))
        self.OriginButton = QtWidgets.QRadioButton(self.centralwidget)
        self.OriginButton.setGeometry(QtCore.QRect(815, 100, 100, 50))
        self.OriginButton.setObjectName("OButton")
        self.XButton = QtWidgets.QRadioButton(self.centralwidget)
        self.XButton.setGeometry(QtCore.QRect(915, 100, 100, 50))
        self.XButton.setObjectName("XButton")
        self.YButton = QtWidgets.QRadioButton(self.centralwidget)
        self.YButton.setGeometry(QtCore.QRect(1015, 100, 100, 50))
        self.YButton.setObjectName("YButton")
        self.ZButton = QtWidgets.QRadioButton(self.centralwidget)
        self.ZButton.setGeometry(QtCore.QRect(1115, 100, 100, 50))
        self.ZButton.setObjectName("ZButton")
        # choose a ground point
        self.label_note = QtWidgets.QLabel(self.centralwidget)
        self.label_note.setGeometry(QtCore.QRect(820, 150, 430, 50))
        self.label_x = QtWidgets.QLabel(self.centralwidget)
        self.label_x.setGeometry(QtCore.QRect(820, 180, 430, 50))
        self.xLable = QtWidgets.QLabel(self.centralwidget)
        self.xLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.xLable.setGeometry(QtCore.QRect(1000, 190, 245, 29))
        self.label_y = QtWidgets.QLabel(self.centralwidget)
        self.label_y.setGeometry(QtCore.QRect(820, 210, 430, 50))
        self.yLable = QtWidgets.QLabel(self.centralwidget)
        self.yLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.yLable.setGeometry(QtCore.QRect(1000, 220, 245, 29))
        # choose a ground point
        self.label_xx = QtWidgets.QLabel(self.centralwidget)
        self.label_xx.setGeometry(QtCore.QRect(820, 240, 430, 50))
        self.xxLable = QtWidgets.QLabel(self.centralwidget)
        self.xxLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.xxLable.setGeometry(QtCore.QRect(1000, 250, 245, 29))
        self.label_xy = QtWidgets.QLabel(self.centralwidget)
        self.label_xy.setGeometry(QtCore.QRect(820, 270, 430, 50))
        self.xyLable = QtWidgets.QLabel(self.centralwidget)
        self.xyLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.xyLable.setGeometry(QtCore.QRect(1000, 280, 245, 29))
        # choose a ground point
        self.label_yx = QtWidgets.QLabel(self.centralwidget)
        self.label_yx.setGeometry(QtCore.QRect(820, 300, 430, 50))
        self.yxLable = QtWidgets.QLabel(self.centralwidget)
        self.yxLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.yxLable.setGeometry(QtCore.QRect(1000, 310, 245, 29))
        self.label_yy = QtWidgets.QLabel(self.centralwidget)
        self.label_yy.setGeometry(QtCore.QRect(820, 330, 430, 50))
        self.yyLable = QtWidgets.QLabel(self.centralwidget)
        self.yyLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.yyLable.setGeometry(QtCore.QRect(1000, 340, 245, 29))
        # choose a ground point
        self.label_zx = QtWidgets.QLabel(self.centralwidget)
        self.label_zx.setGeometry(QtCore.QRect(820, 360, 430, 50))
        self.zxLable = QtWidgets.QLabel(self.centralwidget)
        self.zxLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.zxLable.setGeometry(QtCore.QRect(1000, 370, 245, 29))
        self.label_zy = QtWidgets.QLabel(self.centralwidget)
        self.label_zy.setGeometry(QtCore.QRect(820, 390, 430, 50))
        self.zyLable = QtWidgets.QLabel(self.centralwidget)
        self.zyLable.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.zyLable.setGeometry(QtCore.QRect(1000, 400, 245, 29))
        # pushButton_Computation
        self.pushButton_Computation = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Computation.setGeometry(QtCore.QRect(815, 560, 430, 50))
        self.pushButton_Computation.setObjectName("pushButton_Computation")
        # label_image
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 800, 600))
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setObjectName("label_image")
        self.label_image.setScaledContents(True)
        # menubar and statusbar
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # show name
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # clicked event
        self.pushButton_openImage.clicked.connect(self.openImage)
        self.pushButton_Computation.clicked.connect(self.calibrationCamera)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", 'Vanishing Point Metrology Computation'))
        self.pushButton_openImage.setText(_translate("MainWindow", "Open Image"))
        self.pushButton_Computation.setText(_translate("MainWindow", "Computation"))        
        self.label_radiobheight.setText(_translate("MainWindow", "Choose point to select:"))
        self.OriginButton.setText(_translate("MainWindow", "O coords"))
        self.XButton.setText(_translate("MainWindow", "X coords"))
        self.YButton.setText(_translate("MainWindow", "Y coords"))
        self.ZButton.setText(_translate("MainWindow", "Z coords"))
        """
        self.heightButton.setText(_translate("MainWindow", "Input Height"))
        self.label_height.setText(_translate("MainWindow", "Camera Height:"))
        """
        self.label_note.setText(_translate("MainWindow", "(Please choose a point as origin point.)"))
        self.label_x.setText(_translate("MainWindow", "Point u in image (px):"))
        self.label_y.setText(_translate("MainWindow", "Point v in image (px):"))
        self.label_xx.setText(_translate("MainWindow", "Point Xx in image (px):"))
        self.label_xy.setText(_translate("MainWindow", "Point Xy in image (px):"))
        self.label_yx.setText(_translate("MainWindow", "Point Yx in image (px):"))
        self.label_yy.setText(_translate("MainWindow", "Point Yy in image (px):"))
        self.label_zx.setText(_translate("MainWindow", "Point Zx in image (px):"))
        self.label_zy.setText(_translate("MainWindow", "Point Zy in image (px):"))

    def openImage(self):  
        global imgName
        global rows_prop
        global cols_prop
        self.OriginButton.toggle()
        imgName = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Choose Image", "", "*.jpg;;*.png;;All Files(*)")[0]
        jpg = QtGui.QPixmap(imgName).scaled(self.label_image.width(), self.label_image.height())
        self.label_image.setPixmap(jpg)
        rows, cols = image_read(imgName)
        rows_prop = rows / self.label_image.height()
        cols_prop = cols / self.label_image.width()
        self.label_image.mousePressEvent = self.getPosPts


    def getPosPts(self, event):
        if (self.OriginButton.isChecked()) :
            global ux
            global vy
            ux = event.pos().x() * cols_prop
            vy = event.pos().y() * rows_prop
            self.getPos(event, self.xLable, self.yLable)
        elif (self.XButton.isChecked()) :
            global pxx
            global pxy
            pxx = event.pos().x() * cols_prop
            pxy = event.pos().y() * rows_prop
            self.getPos(event, self.xxLable, self.xyLable)
        elif (self.YButton.isChecked()) :            
            global pyx
            global pyy
            pyx = event.pos().x() * cols_prop
            pyy = event.pos().y() * rows_prop
            self.getPos(event, self.yxLable, self.yyLable)
        elif (self.ZButton.isChecked()) :            
            global pzx
            global pzy
            pzx = event.pos().x() * cols_prop
            pzy = event.pos().y() * rows_prop
            self.getPos(event, self.zxLable, self.zyLable)
    
    def getPos(self, event, xObject, yObject):
        global gx
        global gy
        global rows_prop
        global cols_prop
        gx = event.pos().x() * cols_prop
        gy = event.pos().y() * rows_prop
        _translate = QtCore.QCoreApplication.translate
        xObject.setText(_translate("MainWindow", str(gx)))
        yObject.setText(_translate("MainWindow", str(gy)))

    
    def calibrationCamera(self):
        global imgName
        global ux, vy, pxx, pxy, pyx, pyy, pzx, pzy
        pu, pv = ux, vy

        p1 = (2315, 2650)
        p2 = (3285, 1820)
        p3 = (2345, 2230)
        p4 = (3315, 1440)
        p5 = (510, 1330)
        p6 = (1535, 645)
        p7 = (505, 1705)
        lines = {
            "x": [[p1, p2], [p3, p4], [p5, p6]],
            "y": [[p1, p3], [p2, p4], [p7, p5]],
            "z": [[p7, p1], [p5, p3], [p6, p4]]
        }
        textures = {
            "x": [[p1, p2], [p3, p4]],
            "y": [[p3, p4], [p5, p6]],
            "z": [[p1, p3], [p7, p5]]
        }

        B = main(imgName, (pu, pv), (pxx, pxy), (pyx, pyy), (pzx, pzy), lines, textures)
        _translate = QtCore.QCoreApplication.translate

def image_read(imgpath):
    image = io.imread(imgName)
    rows = image.shape[0] 
    cols = image.shape[1]
    return rows, cols

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    obj = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(obj)
    obj.show()
    sys.exit(app.exec_())
