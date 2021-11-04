import os, fnmatch
import numpy as np
import scipy.io
import tifffile as tiff
import h5py


class Processed:
    """
    A class used to load and save already processed datasets (or image files) in .mgh,.tif,.h5 or .mat format
    """

    def __init__(self):
        self.nMetaBytes = int(1024 * 1024)
        self.storageType = 'uint8'
        self.directory = None
        self.image = None
        self.filename = None
        self.meta = []
        self.storedData = {}
        self.frame=0



        self.loadStates = {
            'struct': 0,
            'angio': 0,
            'weight': 0,
            'dop': 0,
            'ret': 0,
            'theta': 0,
            }



    def setState(self, state='struct'):
        """
        Set the Loading state for loading of frame range
        """
        for key, val in self.loadStates.items():
            if key in state:
                self.loadStates[key] = 1
            else:
                self.loadStates[key] = 0
        if 'ps' in state:
            self.loadStates['dop'] = 1
            self.loadStates['ret'] = 1
            self.loadStates['theta'] = 1


    def loadMGHMeta(self, fullname):
        """
        Loads metadata from a given file

        Args:
            self (obj) : for storage
            fullname (str) : Full filename of .mgh file from which to load metadata
        """
        try:
            with open(fullname, 'r') as f:
                self.meta = np.fromfile(f, count=7, dtype=np.int32)
        except Exception as e:
            raise e

    def loadMGH(self, directory, frame, fileType):
        """
        Loads individual frames from .MGH data

        Note:
            The Processed directory within the main directory is crawled to find a string designated by the filetype
        Args:
            self (obj) : for storage
            directory (str): directory of processed data
            frame (int) : Full filename of .mgh file from which to load metadata
            fileType (str) : the type of mgh file to load (ie. tomoX, struct, angio)

        """
        self.directory = directory
        self.basename = os.path.join(self.directory, os.path.splitext(os.path.splitext(fnmatch.filter(os.listdir(directory), '*.mgh')[0])[0])[0])
        fileID = ''
        nBytes = 0
        loadScat = 0
        loadType = self.storageType

        if 'tomch1' in fileType:
            nBytes = 4
            loadType = 'float32'
            loadScat = 1
        elif 'tomch2' in fileType:
            nBytes = 4
            loadType = 'float32'
            loadScat = 1
        else:
            nBytes = 1
            loadType = 'uint8'

        fileID = '*' + fileType + '*.mgh'
        filename = fnmatch.filter(os.listdir(self.directory), fileID)
        self.fullname = os.path.join(self.directory, filename[0])

        fileCheck = len(filename)
        if fileCheck == 1:
            self.loadMGHMeta(self.fullname)
            totalFrames = self.meta[4]
            imgWidth = self.meta[2]
            imgHeight = self.meta[3]
            frameSize = np.int64(imgWidth * imgHeight)
            Offset = np.int64(self.nMetaBytes + np.int64(((frame - 1) * frameSize * nBytes)))
            Count = np.int64(frameSize * nBytes)

            if frame <= totalFrames:
                with open(self.fullname, 'r') as f:
                    temp = np.fromfile(f, count=Count, offset=Offset, dtype=loadType)

                if loadScat:
                    realScat = temp[0::2]
                    imagScat = temp[1::2]
                    self.image = (realScat + imagScat * 1j).reshape(imgHeight, imgWidth)
                else:
                    temp = temp.reshape(imgHeight, imgWidth)
                    self.image = temp
                print('MGH frame {} loaded from: {}'.format(frame, filename[0]))
                print('Shape of data: {},{},{}'.format(imgWidth, imgHeight, totalFrames))
            else:
                print(
                    'Frame requested exceeds total frames, there is only {} frames in this file'.format(totalFrames))
        elif fileCheck < 1:
            print('No {} file found, please try another folder'.format(fileType))
        elif fileCheck > 1:
            print('Too many {} files in the directory'.format(fileType))

        return self.image

    def memmapMGH(self, directory, fileType):
        """
        Loads individual frames from .MGH data

        Note:
            The Processed directory within the main directory is crawled to find a string designated by the filetype
        Args:
            self (obj) : for storage
            directory (str): directory of processed data
            frame (int) : Full filename of .mgh file from which to load metadata
            fileType (str) : the type of mgh file to load (ie. tomoX, struct, angio)

        """

        self.directory = directory
        self.basename = os.path.join(self.directory, os.path.splitext(os.path.splitext(fnmatch.filter(os.listdir(directory), '*.mgh')[0])[0])[0])
        fileID = ''
        nBytes = 0
        loadScat = 0
        loadType = self.storageType

        if 'tomch1' in fileType:
            nBytes = 4
            loadType = 'float32'
            loadScat = 1
        elif 'tomch2' in fileType:
            nBytes = 4
            loadType = 'float32'
            loadScat = 1
        else:
            nBytes = 1
            loadType = 'uint8'


        fileID = '*' + fileType + '*.mgh'
        filename = fnmatch.filter(os.listdir(self.directory), fileID)
        self.fullname = os.path.join(self.directory, filename[0])

        self.loadMGHMeta(self.fullname)

        fpr = np.memmap(self.fullname, dtype='uint8', mode='r', offset=1024 * 1024,
                        shape=(self.meta[2], self.meta[3], self.meta[4]), order='F')

        return fpr


    def loadNpy(self, filename):
        """Loads .mat files into a dict

        Args:
            filename: *.mat filename
        Returns:
            matData
        """
        return np.load(filename)

    def saveNpy(self, filename, mat):
        """Loads .mat files into a dict

        Args:
            filename: *.mat filename
        Returns:
            matData
        """
        return np.save(filename, mat)

    def loadMat(self, filename):
        """Loads .mat files into a dict

        Args:
            filename: *.mat filename
        Returns:
            matData
        """
        return scipy.io.loadmat(filename)

    def writeMat(self,filename, varName, mat):
        """ Write numpy array to .mat

        Args:
            filename: Output filename
            varName: MATLAB variable name
            mat: numpy array
        """
        scipy.io.savemat(filename, mdict={varName: mat})

    def loadTIF(self,filename):
        """ Load .tiff images

        Args:
            filename: Input filename
        Returns:
            Image dataset
        """
        return tiff.imread(filename)

    def writeTif(self, filename, mat, rgb=False):
        """ Write numpy array to .tif

        Args:
            filename: Output filename
            mat: numpy array

        """
        if rgb:
            tiff.imwrite(filename, mat, photometric='rgb')
        else:
            tiff.imwrite(filename, mat, photometric='minisblack')

    def loadH5(self, filename=None, frame=None, dataStr='dataset'):
        """ Load .h5 images

        Args:
            filename: Input filename
        Returns:
            dataset: Image dataset
        """
        if frame is None:
            frame = 0
        else:
            frame = frame-1

        with h5py.File(filename, 'r') as f:
            dataset = f[dataStr][..., frame]
        return np.swapaxes(dataset,0,1)

    def writeH5(self, filename, mat, dataStr='dataset', dType='uint8'):
        """ Write numpy array to .mat
        Args:
            filename: Output filename
            dataStr: Dataset name
            mat: numpy array

        """
        with h5py.File(filename, 'w') as f:
            f.create_dataset(dataStr, data=mat, dtype=dType)

    def loadPstif(self,filename):
        """ Load .tiff images

        Args:
            filename: Input filename
        Returns:
            Image dataset
        """
        return tiff.imread(filename)

    def writePstif(self, filename, mat, rgb=False):
        """ Write numpy array to .tif

        Args:
            filename: Output filename
            mat: numpy array

        """
        if rgb:
            tiff.imwrite(filename, mat, photometric='rgb')
        else:
            tiff.imwrite(filename, mat, photometric='minisblack')

    def memmapTiff(self, filename):
        """
        Memory maps .tiff data
        """

        return tiff.memmap(filename)