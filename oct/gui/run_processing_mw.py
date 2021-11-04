# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui',
# licensing of 'mainwindow.ui' applies.
#
# Created: Fri Mar  6 08:12:30 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("oct-cbort")
        MainWindow.resize(600, 320)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.cbox_runBatch = QtWidgets.QCheckBox(self.centralwidget)
        self.cbox_runBatch.setGeometry(QtCore.QRect(340, 10, 201, 30))
        self.cbox_runBatch.setObjectName("cbox_runBatch")

        self.button_selectDir = QtWidgets.QPushButton(self.centralwidget)
        self.button_selectDir.setGeometry(QtCore.QRect(10, 40, 130, 30))
        self.button_selectDir.setAutoDefault(False)
        self.button_selectDir.setObjectName("button_selectDir")


        self.label_directory = QtWidgets.QLabel(self.centralwidget)
        self.label_directory.setGeometry(QtCore.QRect(10, 10, 310, 30))
        self.label_directory.setObjectName("label_directory")

        self.button_reloadDir = QtWidgets.QPushButton(self.centralwidget)
        self.button_reloadDir.setGeometry(QtCore.QRect(460, 40, 130, 30))
        self.button_reloadDir.setAutoDefault(False)
        self.button_reloadDir.setObjectName("button_reloadDir")

        self.button_ofdViewer = QtWidgets.QPushButton(self.centralwidget)
        self.button_ofdViewer.setGeometry(QtCore.QRect(460, 140, 130, 30))
        self.button_ofdViewer.setAutoDefault(False)
        self.button_ofdViewer.setObjectName("button_ofdViewer")

        self.dirString = QtWidgets.QLineEdit(self.centralwidget)
        self.dirString.setGeometry(QtCore.QRect(150, 40, 300, 30))
        self.dirString.setObjectName("dirString")

        self.label_state= QtWidgets.QLabel(self.centralwidget)
        self.label_state.setGeometry(QtCore.QRect(10, 80, 291, 30))
        self.label_state.setObjectName("label_process")

        self.stateString = QtWidgets.QLineEdit(self.centralwidget)
        self.stateString.setGeometry(QtCore.QRect(20, 120, 150, 30))
        self.stateString.setText('struct+angio+ps')
        self.stateString.setObjectName("stateString")


        self.button_checkFrame = QtWidgets.QPushButton(self.centralwidget)
        self.button_checkFrame.setGeometry(QtCore.QRect(10, 170, 170, 30))
        self.button_checkFrame.setObjectName("button_checkFrame")

        self.input_singleFrame = QtWidgets.QTextEdit(self.centralwidget)
        self.input_singleFrame.setGeometry(QtCore.QRect(200, 170, 50, 30))
        self.input_singleFrame.setObjectName("input_singleFrame")


        self.label_process = QtWidgets.QLabel(self.centralwidget)
        self.label_process.setGeometry(QtCore.QRect(10, 210, 310, 30))
        self.label_process.setObjectName("label_process")

        self.button_processFrameRange = QtWidgets.QPushButton(self.centralwidget)
        self.button_processFrameRange.setGeometry(QtCore.QRect(10, 245, 170, 30))
        self.button_processFrameRange.setObjectName("button_processFrameRange")


        self.input_startFrame = QtWidgets.QTextEdit(self.centralwidget)
        self.input_startFrame.setGeometry(QtCore.QRect(200, 245, 50, 30))
        self.input_startFrame.setObjectName("input_singleFrame")

        self.label_to = QtWidgets.QLabel(self.centralwidget)
        self.label_to.setGeometry(QtCore.QRect(260, 245, 30, 30))
        self.label_to.setObjectName("label_to")

        self.input_endFrame = QtWidgets.QTextEdit(self.centralwidget)
        self.input_endFrame.setGeometry(QtCore.QRect(300, 245, 50, 30))
        self.input_endFrame.setObjectName("input_endFrame")

        self.button_procAllFrames = QtWidgets.QPushButton(self.centralwidget)
        self.button_procAllFrames.setGeometry(QtCore.QRect(370, 245, 170, 30))
        self.button_procAllFrames.setObjectName("button_procAllFrames")


        # self.line = QtWidgets.QFrame(self.centralwidget)
        # self.line.setGeometry(QtCore.QRect(340, 90, 20, 581))
        # self.line.setFrameShape(QtWidgets.QFrame.VLine)
        # self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line.setObjectName("line")
        #

        #
        # self.line_2 = QtWidgets.QFrame(self.centralwidget)
        # self.line_2.setGeometry(QtCore.QRect(130, 83, 221, 16))
        # self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        # self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_2.setObjectName("line_2")
        #

        self.label_File_Formats = QtWidgets.QLabel(self.centralwidget)
        self.label_File_Formats.setGeometry(QtCore.QRect(300, 80, 291, 30))
        self.label_File_Formats.setObjectName("label_File_Formats")

        self.cbox_MGHOUT = QtWidgets.QCheckBox(self.centralwidget)
        self.cbox_MGHOUT.setGeometry(QtCore.QRect(310, 110, 80, 20))
        self.cbox_MGHOUT.setObjectName("cbox_H5OUT")
        self.cbox_MGHOUT.setChecked(True)

        self.cbox_H5OUT = QtWidgets.QCheckBox(self.centralwidget)
        self.cbox_H5OUT.setGeometry(QtCore.QRect(310, 130, 80, 20))
        self.cbox_H5OUT.setObjectName("cbox_H5OUT")

        self.cbox_TIFFOUT = QtWidgets.QCheckBox(self.centralwidget)
        self.cbox_TIFFOUT.setGeometry(QtCore.QRect(310, 150, 80, 20))
        self.cbox_TIFFOUT.setObjectName("cbox_H5OUT")



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar()
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1096, 22))
        self.menubar.setObjectName("menubar")
        self.menuOFDI_Processing = QtWidgets.QMenu(self.menubar)
        self.menuOFDI_Processing.setObjectName("menuOFDI_Processing")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuOFDI_Processing.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "oct-cbort", None, -1))
        self.button_selectDir.setText(QtWidgets.QApplication.translate("MainWindow", "Select Directory", None, -1))
        self.button_reloadDir.setText(QtWidgets.QApplication.translate("MainWindow", "Re-load Directory", None, -1))
        self.button_ofdViewer.setText(QtWidgets.QApplication.translate("MainWindow", "Open .ofd viewer", None, -1))

        self.button_checkFrame.setText(QtWidgets.QApplication.translate("MainWindow", "Check Single Frame", None, -1))
        self.button_processFrameRange.setText(QtWidgets.QApplication.translate("MainWindow", "Process Frame Range", None, -1))
        self.input_singleFrame.setHtml(QtWidgets.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None, -1))
        self.input_startFrame.setHtml(QtWidgets.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None, -1))
        self.label_to.setText(QtWidgets.QApplication.translate("MainWindow", "to", None, -1))
        self.input_endFrame.setHtml(QtWidgets.QApplication.translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None, -1))
        self.label_directory.setText(QtWidgets.QApplication.translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Choose parent directory of data: </span></p></body></html>", None, -1))
        self.label_state.setText(QtWidgets.QApplication.translate("MainWindow","<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Type process state:</span></p></body></html>",None, -1))
        self.label_process.setText(QtWidgets.QApplication.translate("MainWindow","<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Choose frames:</span></p></body></html>",None, -1))
        self.label_File_Formats.setText(QtWidgets.QApplication.translate("MainWindow","<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Choose output filetypes:</span></p></body></html>",None, -1))

        self.button_procAllFrames.setText(QtWidgets.QApplication.translate("MainWindow", "Process All Frames", None, -1))
        self.cbox_runBatch.setText(QtWidgets.QApplication.translate("MainWindow", "Run on Batch Directory  (NF)", None, -1))
        self.cbox_H5OUT.setText(QtWidgets.QApplication.translate("MainWindow", "*.H5", None, -1))
        self.cbox_TIFFOUT.setText(QtWidgets.QApplication.translate("MainWindow", "*.tiff", None, -1))
        self.cbox_MGHOUT.setText(QtWidgets.QApplication.translate("MainWindow", "*.mgh", None, -1))
        self.menuOFDI_Processing.setTitle(QtWidgets.QApplication.translate("MainWindow", "OFDI Processing", None, -1))

from pyqtgraph import GraphicsLayoutWidget
