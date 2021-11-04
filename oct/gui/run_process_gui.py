from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLineEdit
from PyQt5.QtCore import pyqtSlot
from .run_processing_mw  import Ui_MainWindow
import pyqtgraph as pg
from oct.reconstruct import *
from oct.post import *
from oct.load import *
from oct.view.ofd import *
import logging


class MainWindow(QMainWindow):
    """ Main window of simple processing dispatcher gui """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.title = 'OFDI Processing V0.1'
        self.directoryChosen = 0

        self.ui.button_selectDir.clicked.connect(self.chooseDirectory)
        self.ui.button_reloadDir.clicked.connect(self.reloadDirectory)
        self.ui.button_ofdViewer.clicked.connect(self.openOfdViewer)
        self.ui.button_checkFrame.clicked.connect(self.processSingleFrame)
        self.ui.button_processFrameRange.clicked.connect(self.processFrameRange)
        self.ui.button_procAllFrames.clicked.connect(self.processAllFrames)


    @pyqtSlot()
    def chooseDirectory(self):
        """Choose and load data from a directory with an *.ofd file and its metadata"""
        dialog = QFileDialog()
        self.dir = dialog.getExistingDirectory(self, "Open Directory", "./",
                                               QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)


        self.ui.dirString.setText(self.dir)
        self.state = 'struct+angio'
        self.data = Load(directory=self.dir)
        self.processer = Post()
        self.directoryChosen = 1
        logging.info('Directory: {}'.format(self.dir))
        logging.info('Processing state: {}'.format(self.state))

    @pyqtSlot()
    def reloadDirectory(self):
        """Loads data from a directory with an *.ofd file and its metadata"""
        self.dir = str(self.ui.dirString.text())
        self.currentFrame = int(self.ui.input_singleFrame.toPlainText())
        self.state = str(self.ui.stateString.text())
        self.data = Load(directory=self.dir)
        self.processer = Post()
        self.directoryChosen = 1
        logging.info('Directory: {}'.format(self.dir))
        logging.info('Processing state: {}'.format(self.state))

    @pyqtSlot()
    def openOfdViewer(self):
        """Opens the napari ofd viewer"""
        self.dir = str(self.ui.dirString.text())
        self.currentFrame = int(self.ui.input_singleFrame.toPlainText())
        self.state = str(self.ui.stateString.text())
        self.data = Load(directory=self.dir)
        self.processer = Post()
        self.directoryChosen = 1
        viewer = OFDView(self.dir, self.state)
        viewer.run()


    @pyqtSlot()
    def processSingleFrame(self):
        """ Process a single frame for inspection"""
        self.dir = str(self.ui.dirString.text())
        self.currentFrame = int(self.ui.input_singleFrame.toPlainText())
        self.state = str(self.ui.stateString.text())
        self.data = Load(directory=self.dir)
        print('You chose frame {}'.format(self.currentFrame))
        print(self.directoryChosen)

        if self.directoryChosen:
            # Temp addition
            self.processer.processFrameRange(self.data, startFrame=self.currentFrame, endFrame=self.currentFrame,
                                        procState=self.state, procAll=False, writeState=False)

            showProcessedData(self.data, self.processer)
            plt.show()

        else:
            self.msgBox = QMessageBox(self)
            self.msgBox.setText("Please choose a directory with OFDI files in it")
            self.msgBox.exec_()

    @pyqtSlot()
    def processFrameRange(self):
        """ Call the post.processFrameRange method given designated frame gui inputs"""
        self.startFrame = int(self.ui.input_startFrame.toPlainText())
        self.endFrame = int(self.ui.input_endFrame.toPlainText())
        self.state = str(self.ui.stateString.text())
        print('You chose frame {}-{}'.format(self.startFrame, self.endFrame))
        print(self.directoryChosen)

        if self.directoryChosen:
            # Temp addition
            self.processer.processFrameRange(self.data, startFrame=self.startFrame, endFrame=self.endFrame,
                                        procState=self.state, procAll=False, writeState=True)


        else:
            self.msgBox = QMessageBox(self)
            self.msgBox.setText("Please choose a directory with OFDI files in it")
            self.msgBox.exec_()

    @pyqtSlot()
    def processAllFrames(self):
        """ Call the post.processFrameRange method all frames input"""
        print('You chose frame all frames')
        print(self.directoryChosen)

        if self.directoryChosen:
            # Temp addition
            self.processer.processFrameRange(self.data, startFrame=1, endFrame=1,
                                        procState=self.state, procAll=True, writeState=True)

        else:
            self.msgBox = QMessageBox(self)
            self.msgBox.setText("Please choose a directory with OFDI files in it")
            self.msgBox.exec_()



class ConsolePanelHandler(logging.Handler):
    """ Reassign the logger to the cmd/terminal for viewing the log in real-time."""
    def __init__(self, parent):
        logging.Handler.__init__(self)
        self.parent = parent

    def emit(self, record):
        self.parent.write(self.format(record))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    logging.basicConfig(level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # logging.info('Directory: {}'.format(sys.argv[1]))
    # logging.info('Processing state: {}'.format(sys.argv[2]))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())