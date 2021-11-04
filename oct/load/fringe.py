import os, cdflib, logging
from ..utils import *
cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

class Fringe:
    """ Load fringe data from a .ofd file """
    def __init__(self, mode = '2channel'):
        self.mode = mode
        self.ch1 = []
        self.ch2 = []

        self.frameBytes = 0
        self.numframes = 0
        self.startBytePos = 0
        self.endBytePos = 0

        self.startFrame = 1
        self.currentFrame = 1
        self.endFrame = 1
        self.numFramesToProc = self.endFrame - self.startFrame + 1
        self.frames = cp.arange(self.startFrame, self.endFrame)

    def loadFringe(self, frame=1, filename=None):
        """ Load a fringe frame from the .ofd file within a directory
        Notes:
             Can be used standalone if a filename is passed.
        Args:
            frame (int): Frame to be loaded
            filename (str): Optional filename to load
        """
        self.currentFrame = np.int(frame)
        if filename:
            ofdPath = filename
        else:
            ofdPath = os.path.join(self.directory, self.basename +'.ofd')

        self.getOfdInfo(ofdPath)

        if self.mode == '2channel':
            self.load2Channel()
        if self.mode == '1channel':
            self.load1Channel()
        if self.mode == 'mmap2channel':
            self.mmap2channel()
        if self.mode == 'mmap1channel':
            self.mmap1channel()

    def loadVolume(self):
        pass

    def getOfdInfo(self, ofdPath):
        """ Get ofd file size information"""
        self.inputFilenames['ofd'] = ofdPath
        self.scanSettings['frameSizeBytes'] = cp.int64(self.reconstructionSettings['numSamples'] *
                                                       self.scanSettings['numAlinesPerRawFrame'] * 2)
        self.totalBytes = cp.int64(os.stat(self.inputFilenames['ofd']).st_size)
        self.numFrames = cp.int64(self.totalBytes / self.scanSettings['frameSizeBytes'])
        self.startBytePos = cp.int64(self.scanSettings['frameSizeBytes'] * (self.currentFrame - 1))
        self.endBytePos = cp.int64(self.scanSettings['frameSizeBytes'] * self.currentFrame)


    def outputSingleFrame(self, frame=None):
        """ Write a single .ofd frame to the directory"""
        if frame is None:
            frame = 1

        self.loadFringe(frame=frame)

        outputName = self.inputFilenames['ofd'][:-4] + '_single_f' + str(frame) + '.ofd'

        with open(outputName, 'wb') as f:
            self.rawBScan.astype('uint16').tofile(f)

    def load2Channel(self):
        """Loads image data from .ofd file with 2 detection channels"""
        if self.endBytePos <= self.totalBytes:
            with open(self.inputFilenames['ofd'], 'rb') as f:
                self.rawBScan = cp.fromfile(f, count=self.scanSettings['frameSizeBytes'], offset=self.startBytePos * 2,
                                            dtype='uint16').copy()
            rawX = cp.reshape(self.rawBScan[0::2], (self.reconstructionSettings['numSamples'],
                                                    self.scanSettings['numAlinesPerRawFrame']), order="F")
            rawY = cp.reshape(self.rawBScan[1::2], (self.reconstructionSettings['numSamples'],
                                                    self.scanSettings['numAlinesPerRawFrame']), order="F")
            self.ch1 = rawX[:, self.scanSettings['AlinesToProcTomo']].astype(cp.int)
            self.ch2 = rawY[:, self.scanSettings['AlinesToProcTomo']].astype(cp.int)

        else:
            logging.warning('End of frame byte location: {} , Total number of bytes: {}'.format(self.totalBytes /
                                                                                                self.endBytePos))

    def load1Channel(self):
        pass

    def mmap2Channel(self):
        """Memory maps volume  .ofd file with 2 detection channels"""
        self.memmap = np.memmap(self.inputFilenames['ofd'],
                        dtype='uint16',
                        mode='r',
                        offset=0,
                        shape=(
                        2, self.reconstructionSettings['numSamples'], self.scanSettings['numAlinesPerRawFrame'],
                        self.scanSettings['numFrames']), order='F')

    def mmap1Channel(self):
        pass
