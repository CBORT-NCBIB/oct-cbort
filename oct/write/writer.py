import h5py
import tifffile as tiff
import importlib.util
from ..utils import *

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

class Writer:
    """
    A writing object for saving processed images as they are reconstructed
    """
    def __init__(self):
        # For MGH
        self.metaLengthbytes = 1048576  # 1024*1024 = 1024 kBytes
        self.metaLength = int(self.metaLengthbytes / 4)  # (int32 = 4bytes)
        self.metadata = []
        # For H5
        self.newFile = True
        self.existingDatasets = ''

        # MGH filetype is great for uint8 data, for the rest it remains complicated, as we have special cases
        self.stokesCase = 'sv1+sv2'
        self.oaCase = 'oa'
        self.hsvCase = 'hsv'
        self.complexCase = 'tomch1+tomch2+k1+k2'
        self.specialCase = self.stokesCase + '+' + self.oaCase + '+' + self.hsvCase + '+' + self.complexCase

    def writeMetadata(self, data, frameType, filenameIn=None):
        """
        Handles the writing of metadata for .mgh outputs prior to beginning reconstruction
        Args:
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            filenameIn (str) : filename for storage
        """

        if 'mgh' in data.storageSettings['storageFileType']:
            self.writeMghMeta(data, frameType, filenameIn=filenameIn)


    def appendImage(self, image, data, frameType, filenameIn=None):
        """
        Handles the appending of  single reconstructed image
        Args:
            image (array) : Processed image to write
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            filenameIn (str) : filename for storage
        """
        if filenameIn is None:
            filenameIn = data.basenameOutPath + '.' + frameType

        if frameType in self.specialCase:
            self.appendSpecialCase(image, data, frameType, filenameIn)
        else:
            self.appendRealImage(image, data, frameType, filenameIn)


    def writeMghMeta(self, data, frameType, filenameIn=None):
        """
        Writes of metadata for .mgh outputs prior to beginning reconstruction
        Args:
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            imgWidth (int) : width of image to be saved (critical for loading into imagej)
            complexData (bool or int) : complex data flag for tomogram writing
            filenameIn (str) : filename for storage
        """
        if filenameIn is None:
            filenameIn = data.storageSettings['basenameOutPath'] + '.' + frameType

        if frameType in self.stokesCase or frameType in self.complexCase:
            storageType = 2
        elif data.storageSettings['storageType'] == 'uint8':
            storageType = 0
        elif data.storageSettings['storageType'] == 'uint16':
            storageType = 1
        elif data.storageSettings['storageType'] == 'float32':
            storageType = 2
        elif data.storageSettings['storageType'] == 'float64':
            storageType = 3
        else:
            storageType = None
            print('Unknown data type')

        if frameType in self.specialCase:
            pass
        else:
            filename = filenameIn + '.mgh'
            self.metadata = [0,
                             storageType,
                             data.processedData[frameType].shape[1],
                             data.processedData[frameType].shape[0],
                             data.numFramesToProc, data.startFrame, data.endFrame]

            metadata = np.zeros((self.metaLength, 1))
            metadata[0:len(self.metadata), 0] = self.metadata
            with open(filename, 'w+') as f:
                np.array(metadata, dtype=np.int32).tofile(f)


    def appendComplexImage(self, image, data, frameType, filenameIn=None):
        """
        Appends single complex reconstructed tomogram image
        Args:
            image (array) : Processed image to write
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            filenameIn (str) : filename for storage
        """

        storageType = data.storageSettings['storageTypeComplex']

        if filenameIn is None:
            filenameIn = data.storageSettings['basenameOutPath'] + '.' + frameType

        try:
            image = cp.asnumpy(image)
        except:
            image = image

        imgWidth = image.shape[1]
        imgHeight = image.shape[0]

        if 'mgh' in data.storageSettings['storageFileType']:
            filename = filenameIn + '.mgh'
            dataFlattened = np.matrix.flatten(image)
            array2write = np.zeros((2 * len(dataFlattened)))
            array2write[0::2] = np.real(dataFlattened)
            array2write[1::2] = np.imag(dataFlattened)
            with open(filename, 'a') as f:
                np.array(array2write, dtype=storageType).tofile(f)

        if 'tif' in data.storageSettings['storageFileType']:
            filename = filenameIn + '.tif'
            if data.currentFrame == data.startFrame:
                with tiff.TiffWriter(filename, append=False, bigtiff=True) as tif:
                    tif.save(np.real(image).astype(storageType))
                    tif.save(np.imag(image).astype(storageType))
            else:
                with tiff.TiffWriter(filename, append=True, bigtiff=True) as tif:
                    tif.save(np.real(image).astype(storageType))
                    tif.save(np.imag(image).astype(storageType))

        if 'h5' in data.storageSettings['storageFileType']:
            filename = filenameIn + '.h5'
            if data.currentFrame == data.startFrame:
                hf = h5py.File(filename, 'w')
                hf.create_dataset('dataset', (imgWidth, imgHeight, data.numFramesToProc, 2),
                                  maxshape=(imgWidth, imgHeight, data.numFramesToProc, 2),
                                  dtype=storageType)
                hf.close()
            with h5py.File(filename, 'a') as f:
                f['dataset'][:, :, (data.currentFrame - data.startFrame), 0] = np.transpose(np.real(image)).astype(
                    storageType)
                f['dataset'][:, :, (data.currentFrame - data.startFrame), 1] = np.transpose(np.imag(image)).astype(
                    storageType)


    def appendRealImage(self, image, data, frameType, filenameIn=None):
        """
        Appends single real reconstructed images to the file formats designated in data.storageSettings['storageFileType'] (str)
        Args:
            image (array) : dictionary of all processed images
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            filenameIn (str) : filename for storage
        """


        image = cp.asnumpy(image)
        storageType = data.storageSettings['storageType']
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]

        if filenameIn is None:
            filenameIn = data.storageSettings['basenameOutPath'] + '.' + frameType

        if 'mgh' in data.storageSettings['storageFileType']:
            filename = filenameIn + '.mgh'
            array2write = np.matrix.flatten(image).astype(storageType)
            with open(filename, 'a') as f:
                np.array(array2write, dtype=storageType).tofile(f)

        if 'tif' in data.storageSettings['storageFileType']:
            filename = filenameIn + '.tif'
            if data.currentFrame == data.startFrame:
                with tiff.TiffWriter(filename, append=False, bigtiff=True) as tif:
                    tif.save(image.astype(storageType))
            else:
                with tiff.TiffWriter(filename, append=True, bigtiff=True) as tif:
                    tif.save(image.astype(storageType))

        if 'h5' in data.storageSettings['storageFileType']:
            if data.storageSettings['mergeH5']:
                filename = data.storageSettings['basenameOutPath'] + '.merge' + '.h5'
                if self.newFile:
                    hf = h5py.File(filename, 'w')
                    hf.create_dataset(frameType, (imgWidth, imgHeight, data.numFramesToProc),
                                      maxshape=(imgWidth, imgHeight, data.numFramesToProc),
                                      dtype=storageType)
                    hf.close()
                    self.newFile = False
                    self.existingDatasets = self.existingDatasets + frameType + '+'

                if frameType not in self.existingDatasets:
                    hf = h5py.File(filename, 'a')
                    hf.create_dataset(frameType, (imgWidth, imgHeight, data.numFramesToProc),
                                      maxshape=(imgWidth, imgHeight, data.numFramesToProc),
                                      dtype=storageType)
                    # hf.close()
                    self.existingDatasets = self.existingDatasets + frameType + '+'

                with h5py.File(filename, 'a') as f:
                    f[frameType][:, :, (data.currentFrame - data.startFrame)] = np.transpose(image).astype(
                        storageType)
            else:
                filename = filenameIn + '.h5'
                if data.currentFrame == data.startFrame:
                    hf = h5py.File(filename, 'w')
                    hf.create_dataset(frameType, (imgWidth, imgHeight, data.numFramesToProc),
                                      maxshape=(imgWidth, imgHeight, data.numFramesToProc),
                                      dtype=storageType)
                    hf.close()
                with h5py.File(filename, 'a') as f:
                    f[frameType][:, :, (data.currentFrame - data.startFrame)] = np.transpose(image).astype(
                        storageType)


    def appendSpecialCase(self, image, data, frameType, filenameIn=None):
        """
        Appends single real reconstructed images to the file formats designated in data.storageSettings['storageFileType'] (str)
        Args:
            image (array) : dictionary of all processed images
            data (object) : data information object
            framteType (str) : type of image frame, ie. 'struct'
            filenameIn (str) : filename for storage
        """


        image = cp.asnumpy(image)
        storageType = data.storageSettings['storageType']
        imgHeight = image.shape[0]
        imgWidth = image.shape[1]

        if filenameIn is None:
            filenameIn = data.storageSettings['basenameOutPath'] + '.' + frameType

        if frameType in self.stokesCase:
            filename = filenameIn + '.h5'
            storageType = 'float32'
            if data.currentFrame == data.startFrame:
                hf = h5py.File(filename, 'w')
                hf.create_dataset(frameType,
                                  (imgWidth, imgHeight, image.shape[2], image.shape[3], data.numFramesToProc),
                                  maxshape=(imgWidth, imgHeight, image.shape[2], image.shape[3], data.numFramesToProc),
                                  dtype=storageType)
                hf.close()

            with h5py.File(filename, 'a') as f:
                out = image.transpose(1, 0, 2, 3)
                f[frameType][:, :, :, :, (data.currentFrame - data.startFrame)] = out

        elif frameType in self.oaCase:
            filename = filenameIn + '.h5'
            storageType = 'float32'
            if data.currentFrame == data.startFrame:
                hf = h5py.File(filename, 'w')
                hf.create_dataset(frameType,
                                  (imgWidth, imgHeight, image.shape[2], data.numFramesToProc),
                                  maxshape=(imgWidth, imgHeight, image.shape[2], data.numFramesToProc),
                                  dtype=storageType)
                hf.close()

            with h5py.File(filename, 'a') as f:
                out = image.transpose(1, 0, 2)
                f[frameType][:, :, :, (data.currentFrame - data.startFrame)] = out


        elif frameType in self.hsvCase:
            filename = filenameIn + '.tif'
            if data.currentFrame == data.startFrame:
                with tiff.TiffWriter(filename, append=False, bigtiff=True) as tif:
                    tif.save(image.astype(storageType))
            else:
                with tiff.TiffWriter(filename, append=True, bigtiff=True) as tif:
                    tif.save(image.astype(storageType))

        elif frameType in self.complexCase:
            filename = filenameIn + '.h5'
            storageType = 'complex64'
            if data.currentFrame == data.startFrame:
                hf = h5py.File(filename, 'w')
                hf.create_dataset(frameType, (imgWidth, imgHeight, data.numFramesToProc),
                                  maxshape=(imgWidth, imgHeight, data.numFramesToProc),
                                  dtype=storageType)
                hf.close()
            with h5py.File(filename, 'a') as f:
                out = image.transpose(1, 0)
                f[frameType][:, :, (data.currentFrame - data.startFrame)] = out


    def saveImage(self, image, filenameIn):
        """
        Saves a single image to the filename designated
        Notes:
            This is mainly for saving projections or single frame outputs for plots.
        Args:
            image (numpy array) : Image to be saved
            filenameIn (str) : Filename for storage
        """
        filename = filenameIn + '.tif'
        with tiff.TiffWriter(filename, append=False, bigtiff=False) as tif:
            tif.save(image.astype('uint16'))
