import sys, os, time
from oct import *
import logging
from PyQt5.QtWidgets import QApplication


class processHandler():
    """ Handles the argument cases for cmd line automated processing"""
    def __init__(self):
        self.directory = ''
        self.state = ''
        self.toggleRange = 1
        self.startFrame = 1
        self.endFrame = 1
        self.status = 0
        self.reconstructID = None
        self.procPath = ''
        self.errorIndicationFile = ''
        self.processingIndicationFile = ''
        self.completeIndicationFile = ''

    def reportProcessingStatus(self):
        """
        Writes an empty reconstruction status file to allow services to track reconstruction completion

        Note:
            Either a reconstruction.complete or a reconstruction.error file will be written depending on reconstruction success

        """
        self.procPath = os.path.join(self.directory, 'Processed')
        if not os.path.exists(self.procPath):
            os.mkdir(self.procPath)

        self.errorIndicationFile = os.path.join(self.procPath, 'process.error')
        self.processingIndicationFile = os.path.join(self.procPath, 'process.processing')
        self.completeIndicationFile = os.path.join(self.procPath, 'process.complete')

        if 'reconstruction' in self.status:
            try:
                os.remove(self.errorIndicationFile)
            except OSError:
                pass
            try:
                os.remove(self.completeIndicationFile)
            except OSError:
                pass
            finally:
                indicatorFile = self.processingIndicationFile
        if 'error' in self.status:
            try:
                os.remove(self.completeIndicationFile)
            except OSError:
                pass
            try:
                os.remove(self.processingIndicationFile)
            except OSError:
                pass
            finally:
                indicatorFile = self.errorIndicationFile
        if 'complete' in self.status:
            try:
                os.remove(self.processingIndicationFile)
            except OSError:
                pass
            try:
                os.remove(self.errorIndicationFile)
            except OSError:
                pass
            finally:
                indicatorFile = self.completeIndicationFile
        with open(indicatorFile, "w") as f:  # Opens file and casts as f
            f.write('success={}'.format(self.status))

    def getInput(self, argv):
        # Directory
        limit = 6
        if len(argv) <= limit:
            print('{} ARGUEMENTS PASSED'.format(len(argv)))
        else:
            print('Too many arguements passed. Max number is {}'.format(limit))
            sys.exit(1)

        try:
            self.directory = str(argv[1])
            print('DIRECTORY   : {}'.format(self.directory))
        except:
            print('Please pass directory_name')
            sys.exit(1)
        # Processing State
        try:
            self.state = str(argv[2])
            print('STATE       : {}'.format(self.state))
        except:
            print('Please pass valid state argument for reconstruction (ie."tomo+struct")')
            sys.exit(1)
        # Output State
        try:
            self.outputType = str(argv[3])
            print('OUTPUT      : {}'.format(self.outputType))
        except:
            print('Please pass valid state argument for reconstruction (ie."mgh+tif")')
            sys.exit(1)
        # File range
        try:
            if str(argv[4]) == 'all':
                self.toggleRange = 'all'
                print('FRAME RANGE : {}'.format('All'))
            else:
                if '-' in argv[4]:
                    frames = np.asarray(argv[4].split('-')).astype(int)
                    self.startFrame = frames[0]
                    self.endFrame = frames[1]
                    self.toggleRange = 'range'
                else:
                    self.startFrame = int(argv[4])
                    self.endFrame = int(argv[4])
                    self.toggleRange = 'single'
                print('FRAME RANGE : {}'.format(argv[4]))
        except:
            print(
                'Please pass frame range. For whole file, type "all" or specify range with two numbers separated by a space (ie. 1 10)')
            sys.exit(1)
        try:
            if len(sys.argv) == 6:
                self.reconstructID = argv[5]
                print('DLL ID # : {}'.format(self.reconstructID))
        except:
            print('Proceeding without dllsettings ID')

    def launchGui(self):
        """ Launch processing GUI"""
        app = QApplication(sys.argv)

        logging.basicConfig(level=logging.INFO, filemode='w',
                            format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        # logging.info('Directory: {}'.format(sys.argv[1]))
        # logging.info('Processing state: {}'.format(sys.argv[2]))

        window = MainWindow()
        window.show()

        sys.exit(app.exec_())

    def launchCmdProcessing(self):
        """ Launch processing from CMD"""
        self.reportProcessingStatus()
        # START LOGGING
        # ========================================================#
        procPath = os.path.join(self.directory, 'Processed')
        if not os.path.exists(procPath):
            os.mkdir(procPath)

        logFile = os.path.join(procPath, 'processing.log')

        logging.basicConfig(filename=logFile,
                            level=logging.INFO,
                            filemode='w',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        logging.info('Directory: {}'.format(self.directory))
        logging.info('Processing state: {}'.format(self.state))
        # START PROCESSING
        # ========================================================#
        t = time.time()
        data = Load(directory=self.directory, id=self.reconstructID)
        data.storageSettings['storageFileType'] = self.outputType
        data.inputFilenames['reconstructID'] = self.reconstructID

        process = Post()

        if self.toggleRange == 'all':
            procWholeFile = True
        else:
            procWholeFile = False

        process.processFrameRange(data,
                                  startFrame=self.startFrame,
                                  endFrame=self.endFrame,
                                  procState=self.state,
                                  procAll=procWholeFile,
                                  writeState=True)

        elapsed = time.time() - t
        if data.success == 1:
            self.status = 'complete'
            # FINISHED PROCESSING
            # ========================================================#
            print('SUCCESS')
            print('Processing took {} minutes and {} seconds'.format(round(elapsed / 60), round(elapsed % 60, 1)))
            logging.info('Processing took {} minutes and {} seconds'.format(round(elapsed / 60),
                                                                            round(elapsed % 60, 1)))
        else:
            print('ERROR    : {}'.format('See log'))
            self.status = 'error'


if __name__ == "__main__":
    try:
        ph = processHandler()
        ph.status = 'reconstruction'

        if len(sys.argv) < 2:
            ph.launchGui()
        else:
            ph.getInput(sys.argv)
            ph.launchCmdProcessing()
            ph.reportProcessingStatus()

    except Exception as e:
        print('ERROR    : {}'.format('See log'))
        logging.error('Error in __main__ method', exc_info=True)
        ph.status = 'error'
        ph.reportProcessingStatus()
        raise e
    finally:
        sys.exit(1)


