from ..utils import nextPowerOf2, closestDividableNumber, checkForCupy
import logging
from oct.utils.psUtils import stokes, unweaveInputPolarizations
from ..load.metadata import Metadata

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

class Tomogram:
    """ Tomogram reconstruction for OCT """
    def __init__(self, mode='heterodyne'):

        acceptedModes = 'heterodyne+minimal+qpr'
        if mode in acceptedModes:
            self.mode = mode
        else:
            self.mode = 'heterodyne'

        self.initialized = False
        self.chCount = 0
        
        self.ch1 = None
        self.ch2 = None
        self.bg1 = None
        self.bg2 = None
        self.chirp = None
        self.dispersion = None

        self.processedData = {
            'tomch1': None,
            'tomch2': None,
            'k1': None,
            'k2': None,
            'sv1': None,
            'sv2': None
        }

        self.fourierLength = 0
        self.fringeWindow = None
        self.spectralFringeWindows = None

        self.numChunks = 1
        self.fastDownSample = 1
        self.rampDownSample = 1

        meta = Metadata()
        self.settings = meta.reconstructionSettings


    def initialize(self, data=None, settings=None):
        """ Initialize tomogram reconstruction variables """

        if data and settings:
            self.setSettings(data=data, settings=settings)
        elif data:
            self.setSettings(data=data)
        elif settings:
            self.setSettings(settings=settings)

        if data is None and settings is None:
            self.settings['numSamples'] = self.ch1.shape[0]
            self.settings['imgWidth'] = self.ch1.shape[1]

        # Length of Real/Valuable portion of the signal after fourier transform 
        self.fourierLength = int(self.settings['numSamples'] / 2)

        # Assure that the number of output pixels are large enough to fit all the valuable information from the data
        if self.fourierLength > self.settings['numZOut']:
            self.settings['numZOut'] = np.int(nextPowerOf2(self.fourierLength))

        # Output tomogram and stokes Z range (for cropping before contrast algorithms)
        if (self.settings['depthIndex'][1] <= self.settings['depthIndex'][0]) or \
            (self.settings['depthIndex'][1] > self.settings['numZOut']):
            self.settings['depthIndex'][1] = self.settings['numZOut']
        self.indexRange = np.arange(self.settings['depthIndex'][0],
                                    self.settings['depthIndex'][1])

        # Update demodulation index according to possible new numZOut
        if self.mode == 'heterodyne':
            carrier = self.settings['clockRateMHz'] * \
                      float(self.settings['demodSet'][0]) / 2
            normDemodCarrier = carrier / (self.settings['clockRateMHz'] / 2)
            self.settings['demodCarrierIndex'] = round(normDemodCarrier * 0.5 *
                                                       self.settings['numSamples'])
            self.settings['demodReverseIndex'] = round(self.settings['demodCarrierIndex'] *
                                                                     self.settings['numZOut'] /
                                                                     self.settings['numSamples'] * 2)


        # Update chirp, dispersion indexes according to possible new numZOut
        self.interpolateChirp()
        self.interpolateDispersion()

        # Generate the "fringe window" - an ideal gaussian source function that we multiply by the detected fringe
        self.fringeWindow = cp.hanning(self.fourierLength)[:, None]
        # Assign fringe window to spectral fringe window in cases of PS without spectral binning
        self.spectralFringeWindows = self.fringeWindow[:, None]

        #numFrameAlines = self.settings['imgDepth'] * self.settings['imgWidth']
        # self.dataRange = cp.arange(0, numFrameAlines) + int(self.settings['chunkPadding'])

        # If imgDepth is 1, do whole frame, otherwise, chunk by imgDepth
        if self.settings['imgDepth'] == 1:
            self.numChunks = 1 * max(self.settings['factorGPURam'], 1)
        else:
            self.numChunks = self.settings['imgDepth'] * self.settings['factorGPURam']

        # Set up spectral binning windows if spectral binning and ps processing is requested
        if 'ps' in self.settings['processState']:
            if self.settings['spectralBinning']:
                self.createSpectralBins()
            else:
                self.settings['numSpectralWindows'] = 1

            if self.settings['fastProcessing']:
                self.fastDownSample = 2
            else:
                self.fastDownSample = 1

            if self.settings['imgDepth'] == 1:
                self.rampDownSample = 2
                self.fastDownSample = 1
            else:
                self.rampDownSample = 1

            # Set padding for stokes vectors
            #self.settings['chunkPadding'] = 2 * self.settings['xFilter']
            self.settings['chunkPadding'] = 0

        # If imgDepth is 1, do whole frame, otherwise, chunk by imgDepth
        if self.settings['imgDepth'] == 1:
            self.numChunks = 1 * max(self.settings['factorGPURam'], 1)
        else:
            self.numChunks = self.settings['imgDepth'] * self.settings['factorGPURam']


        self.allocateSpace(data)

        self.initialized = True
        logging.info('====================================================')
        logging.info('Tomogram settings initialized')

    def setSettings(self, data=None, settings=None):
        """Extract the required settings variables from the dataset metadata"""

        if data:
            for key, val in data.reconstructionSettings.items():
                self.settings[key] = data.reconstructionSettings[key]

        elif settings:
            for key, val in settings.items():
                self.settings[key] = settings[key]

    def requires(self):
        """
        This method returns the required veriables to perform reconstructions
        """
        print('Minimum Required:')
        print("ch1=ch1")
        print("settings=settings")
        print('\n')
        print("OR")
        print('\n')
        print("data=data (processing pipeline object)")
        print('\n')
        print('Optional:')
        print("ch2=ch2")
        print("bg1=bg2")
        print("bg2=bg2")
        print("chirp=chirp")
        print("dispersion=dispersion")

        print('Required for stokes Vectors:')
        print("settings['processState'] -> if stokes vectors are desired, set to any string with 'ps' in it")
        print("settings['spectralBinning'] -> default: {}".format(self.settings['spectralBinning']))
        print("settings['fastProcessing'] -> default: {}".format(self.settings['fastProcessing']))
        print("settings['binFract'] -> default: {}".format(self.settings['binFract']))
        print("settings['holdOnGPURam'] -> default: {}".format(self.settings['holdOnGPURam']))
        print('\n')
        print('For reference, all possible settings and defaults are:')
        for key, value in self.settings.items():
            print("self.settings['", key, "'] : ", value)

    def interpolateChirp(self, chirp=None):
        """
        Interpolates chirp information for tomogram reconstruction

        Args:
            self (obj) : For self.calibrationSettings['chirpFileCheck'], self.rawChirpData,  self.scanAxis,
            self.reconstructionSettings['zoomFactor']
        """
        if not (chirp is None):
            self.chirp = chirp

        if not (self.chirp is None):
            chirp = cp.asnumpy(self.chirp)
            # Interpolate to Samples
            xchirp = np.arange(0, 1, 1 / len(self.chirp))
            xsample = np.arange(0, 1, 1 / (self.fourierLength))
            interpChirp = np.interp(xsample, xchirp, chirp)
            # Force edges
            interpChirp[0] = 0
            interpChirp[-1] = 1
            # Account for zoom factor
            interpChirp = interpChirp * self.settings['numSamples'] * self.settings['zoomFactor']
            chirp = np.floor(interpChirp)
            chirp[0] = 1
            chirp[-1] = self.settings['numSamples'] * self.settings['zoomFactor'] - 1
        else:
            # Create linear chirp
            chirp = np.floor(
                self.settings['numSamples'] *
                self.settings['zoomFactor'] *
                np.arange(0, 1, 1 / self.settings['numSamples'] * 2))
            chirp[0] = 1
            chirp[-1] = self.settings['numSamples'] * self.settings['zoomFactor'] - 1

        chirpIndex = cp.asarray(np.searchsorted(np.arange(1, self.settings['numSamples'] *
                                               self.settings['zoomFactor'] + 1), chirp))
        logging.info('Chirp interpolated')

        self.chirp = chirpIndex

    def interpolateDispersion(self, dispersion=None):
        """
        Interpolates complexIndex dispersion information for tomogram reconstruction

        Args:

        """
        if not (dispersion is None):
            self.dispersion = dispersion

        if not (self.dispersion is None):
            dispersion = cp.asnumpy(self.dispersion)
            # Process data
            complexIndex = int(len(dispersion) / 2)
            dispersion = dispersion[0:complexIndex] + 1j * dispersion[complexIndex:2 * complexIndex]
            # Interpolate to Samples
            xdispersion = np.arange(0, 1, 1 / complexIndex)
            xsample = np.arange(0, 1, 1 / (self.settings['numSamples'] / 2))
            dispersion = cp.asarray(np.interp(xsample, xdispersion, dispersion))
        else:
            dispersion = cp.ones(np.int(self.settings['numSamples'] / 2), dtype='complex64')
        logging.info('Dispersion data processed')

        self.dispersion = dispersion

    def allocateSpace(self, data):
        """ Initializes GPU memory for chunked tomogram reconstruction"""
        if freeMemory > self.settings['minGPURam']:
            self.holdStokesOnGPU = 1

        # Create storage arrays for chunk processing
        self.processedData['tomch1'] = cp.zeros(
            (len(self.indexRange),
             self.settings['imgWidth'] * self.settings['imgDepth']),
            dtype='complex64')

        if self.chCount == 2:
            self.processedData['tomch2'] = cp.zeros(
                (len(self.indexRange),
                 self.settings['imgWidth'] * self.settings['imgDepth']),
                dtype='complex64')

        if 'kspace' in self.settings['processState']:
            self.processedData['k1'] = cp.zeros(
                (np.int(self.settings['numSamples']/2),
                 self.settings['imgWidth'] * self.settings['imgDepth']),
                dtype='complex64')
            if self.chCount == 2:
                self.processedData['k2'] = cp.zeros(
                    (np.int(self.settings['numSamples']/2),
                     self.settings['imgWidth'] * self.settings['imgDepth']),
                    dtype='complex64')

        if 'ps' in self.settings['processState']:
            # Create storage arrays for chunk processing
            if self.settings['holdOnGPURam']:
                self.processedData['sv1'] = cp.zeros(
                    (len(self.indexRange),
                     int(self.settings['imgWidth'] / self.fastDownSample / self.rampDownSample
                            + 2 * self.settings['chunkPadding']), 4,
                     self.settings['numSpectralWindows']), dtype='float32')
                if self.chCount == 2:
                    self.processedData['sv2'] = cp.zeros(
                        (len(self.indexRange),
                         int(self.settings['imgWidth'] / self.fastDownSample / self.rampDownSample
                                + 2 * self.settings['chunkPadding']), 4,
                         self.settings['numSpectralWindows']), dtype='float32')
                logging.info('Enough VRAM, holding Stokes Vectors on GPU')
                logging.info('Stokes Vector shape:{}'.format(self.processedData['sv1'].shape))
            else:
                self.processedData['sv1'] = np.zeros(
                    (len(self.indexRange),
                     np.int(self.settings['imgWidth'] / self.fastDownSample / self.rampDownSample
                            + 2 * self.settings['chunkPadding']), 4,
                     self.settings['numSpectralWindows']), dtype='float32')
                if self.chCount == 2:
                    self.processedData['sv2'] = np.zeros(
                        (len(self.indexRange),
                         np.int(self.settings['imgWidth'] / self.fastDownSample / self.rampDownSample
                                + 2 * self.settings['chunkPadding']), 4, self.settings['numSpectralWindows']), dtype='float32')
                logging.info('Not enough VRAM, holding Stokes Vectors on CPU')
                logging.info('Stokes Vector shape:{}'.format(self.processedData['sv1'].shape))

    def createSpectralBins(self):
        """ Creates spectral binning windows to subsample fringe data in K-space, after correction """

        self.settings['numSpectralWindows'] = int(self.settings['binFract'] * 2 - 1)
        hanningLength = round(self.fourierLength / self.settings['binFract'])
        hanningWindow = cp.hanning(hanningLength)[:, cp.newaxis]
        zeroPad = cp.zeros((int(self.fourierLength - hanningLength), 1))
        spectralFringeWindows = cp.zeros((int(self.fourierLength), self.settings['numSpectralWindows']))
        num = cp.sum(cp.hanning(self.fourierLength) ** 2)
        den = cp.sum(cp.hanning(hanningLength) ** 2)
        scale = cp.sqrt(num / den)

        for window in range(self.settings['numSpectralWindows']):
            padded = cp.concatenate((hanningWindow, zeroPad), axis=0)
            rollFrac = round((window * self.fourierLength) /
                             (self.settings['numSpectralWindows'] + 1))
            spectralFringeWindows[:, window] = cp.roll(padded, rollFrac)[:, 0] * scale
        self.spectralFringeWindows = cp.expand_dims(spectralFringeWindows, axis=1).astype('float32')

    def manageChannels(self):
        """ Handle case where only channel 2's are passed """
        if not (self.ch2 is None) and (self.ch1 is None):
            self.ch1 = self.ch2
            self.ch2 = None
        if not (self.bg2 is None) and (self.bg1 is None):
            self.bg1 = self.bg2
            self.bg2 = None

    def manageBackground(self):
        """ Handle case where only channel 2's are passed """


        if not (self.bg2 is None) and (self.bg1 is None):
            self.bg1 = self.bg2
            self.bg2 = None

        if (self.bg2 is None) and (self.bg1 is None):
            if (self.ch2 is None):
                self.bgv1 = cp.median(self.ch1[:, 0::2], axis=1)
                self.bgh1 = cp.median(self.ch1[:, 0::2], axis=1)
            else:
                self.bgv1 = cp.median(self.ch1[:, 0::2], axis=1)
                self.bgh1 = cp.median(self.ch1[:, 0::2], axis=1)
                self.bgv2 = cp.median(self.ch2[:, 0::2], axis=1)
                self.bgh2 = cp.median(self.ch2[:, 0::2], axis=1)

        elif self.settings['bgRemoval'] == 'median':
            if not (self.ch2 is None):
                self.bgv1 = cp.median(self.ch1[:, 0::2], axis=1)
                self.bgh1 = cp.median(self.ch1[:, 1::2], axis=1)
                self.bgv2 = cp.median(self.ch2[:, 0::2], axis=1)
                self.bgh2 = cp.median(self.ch2[:, 1::2], axis=1)
            else:
                self.bgv1 = cp.median(self.ch1[:, 0::2], axis=1)
                self.bgh1 = cp.median(self.ch1[:, 1::2], axis=1)


        elif self.settings['bgRemoval'] == 'ofb':
            # Check if polarization channel is included
            if not (self.bg2 is None):
                if len(self.bg1.shape) < 2:
                    self.bgv1 = cp.asarray(self.bg1[0::2])
                    self.bgh1 = cp.asarray(self.bg1[1::2])
                    self.bgv2 = cp.asarray(self.bg2[0::2])
                    self.bgh2 = cp.asarray(self.bg2[1::2])
                else:
                    self.bgv1 = cp.asarray(self.bg1[:, 0])
                    self.bgh1 = cp.asarray(self.bg1[:, 1])
                    self.bgv2 = cp.asarray(self.bg2[:, 0])
                    self.bgh2 = cp.asarray(self.bg2[:, 1])
            else:
                if len(self.bg1.shape) < 2:
                    self.bgv1 = cp.asarray(self.bg1[0::2])
                    self.bgh1 = cp.asarray(self.bg1[1::2])
                else:
                    self.bgv1 = cp.asarray(self.bg1[:, 0])
                    self.bgh1 = cp.asarray(self.bg1[:, 1])

    def reconstruct(self, ch1=None, ch2=None, bg1=None, bg2=None,
                    chirp=None, dispersion=None, data=None, settings=None):
        """ Handles the reconstruction of the tomogram from the data object"""

        if not self.initialized:
            if data:
                self.ch1 = cp.asarray(data.ch1.copy())
                self.ch2 = cp.asarray(data.ch2.copy())
                self.bg1 = cp.asarray(data.bg1.copy())
                self.bg2 = cp.asarray(data.bg2.copy())
                self.chirp = cp.asarray(data.chirp.copy())
                self.dispersion = cp.asarray(data.dispersion.copy())
                self.chCount = 2
            else:
                if not (ch1 is None):
                    self.chCount = self.chCount + 1
                    self.ch1 = cp.asarray(ch1.copy())
                if not (ch2 is None):
                    self.chCount = self.chCount + 1
                    self.ch2 = cp.asarray(ch2.copy())
                if not (bg1 is None):
                    self.bg1 = cp.asarray(bg1.copy())
                if not (bg2 is None):
                    self.bg2 = cp.asarray(bg2.copy())
                if not (chirp is None):
                    self.chirp = cp.asarray(chirp.copy())
                if not (dispersion is None):
                    self.dispersion = cp.asarray(dispersion.copy())

            self.manageChannels()
            self.manageBackground()
            self.initialize(data=data, settings=settings)

        else:
            if data:
                self.ch1 = cp.asarray(data.ch1.copy())
                self.ch2 = cp.asarray(data.ch2.copy())
            else:
                if not (ch1 is None):
                    self.ch1 = cp.asarray(ch1.copy())
                if not (ch2 is None):
                    self.ch2 = cp.asarray(ch2.copy())

        if 'heterodyne' in self.mode:
            if self.chCount == 2:
                self.heterodyne2channel()
            else:
                self.heterodyne1channel()
        if 'minimal' in self.mode:
            if self.chCount == 2:
                self.minimal2channel()
            else:
                self.minimal1channel()

        # for key, val in self.processedData.items():
        #     if not (val is None):
        #         self.processedData[key] = self.processedData[key][self.indexRange]
        #     else:
        #         self.processedData[key] = self.processedData[key]

        return self.processedData

    def heterodyne2channel(self):
        """
        Compute an optical coherence tomogram using heterodyne 2-channel polarization diverse detection

        Note:
            If PS reconstruction enabled, both am unbinned and a binned complex tomogram  will be produced.
            Reference(s):
            Tomogram reconstruction:
            [1] Yun, S., Tearney, G., de Boer, J., Iftimia, N., & Bouma, B. (2003).
                High-speed optical frequency-domain imaging. Optics Express, 11(22), 2953.
                https://doi.org/10.1364/oe.11.002953
            [2] Motaghian Nezam, S. M. R., Vakoc, B. J., Desjardins, A. E., Tearney, G. J., & Bouma, B. E. (2007).
                Increased ranging depth in optical frequency domain imaging by frequency encoding. Optics Letters,
                32(19), 2768.
                https://doi.org/10.1364/ol.32.002768
            Stokes Vector calculation:
            [1] Park BH, Pierce MC, Cense B, de Boer JF.
                Real-time multi-functional optical coherence tomography.
                Opt Express. 2003;11(7):782–793
        Args:
            param1 (obj): data
        Output:
            self.processedData['tomch1'], self.processedData['tomch2'], self.SV1, self,SV2

        """

        # try:
        # Initialize on GPU Memory and remove bg1
        self.processedData['tomch1'][:] = 0
        self.processedData['tomch2'][:] = 0

        self.ch1[:, 0::2] = self.ch1[:, 0::2] - self.bgv1[:, None]
        self.ch1[:, 1::2] = self.ch1[:, 1::2] - self.bgh1[:, None]
        self.ch2[:, 0::2] = self.ch2[:, 0::2] - self.bgv2[:, None]
        self.ch2[:, 1::2] = self.ch2[:, 1::2] - self.bgh2[:, None]

        chunkSize = int(self.ch1.shape[1] / self.numChunks)
        numChunks = int(cp.floor(self.ch1.shape[1] / chunkSize))
        logging.info('Chunk Size : {}'.format(chunkSize))
        logging.info('Number of chunks: {}'.format(numChunks))

        # Grab a little extra to make sure we complete the downsample properly
        chunkSizePadded = closestDividableNumber(chunkSize, self.settings['imgDepth'] * 2)
        chunkSizeDifference = chunkSizePadded - chunkSize
        zoomCh1 = cp.zeros((self.settings['numSamples'] * self.settings['zoomFactor'], chunkSizePadded),
                           dtype='complex64')
        zoomCh2 = cp.zeros((self.settings['numSamples'] * self.settings['zoomFactor'], chunkSizePadded),
                           dtype='complex64')

        for c in range(numChunks):
            arrayRangeGrabbed = np.arange(chunkSize * c, (chunkSize) * (c + 1) + chunkSizeDifference, 1)
            arrayRangePut = np.arange(chunkSize * c, (chunkSize) * (c + 1), 1)

            # Apply first fft
            if c == numChunks-1:
                dataChunkX = cp.zeros((self.ch1.shape[0], chunkSizePadded), dtype='complex64')
                dataChunkY = cp.zeros((self.ch1.shape[0], chunkSizePadded), dtype='complex64')
                dataChunkX[:, 0:len(arrayRangePut)] = cp.fft.fft(self.ch1[:, arrayRangePut], axis=0)
                dataChunkY[:, 0:len(arrayRangePut)] = cp.fft.fft(self.ch2[:, arrayRangePut], axis=0)
            else:
                dataChunkX = cp.fft.fft(self.ch1[:, arrayRangeGrabbed], axis=0)
                dataChunkY = cp.fft.fft(self.ch2[:, arrayRangeGrabbed], axis=0)


            zoomCh1[:] = 0
            zoomCh2[:] = 0

            # Take real data
            zoomCh1[0:self.fourierLength, :] = dataChunkX[0:self.fourierLength, :]
            zoomCh2[0:self.fourierLength, :] = dataChunkY[0:self.fourierLength, :]
            dataChunkX, dataChunkY = None, None
            # Demodulation
            zoomCh1 = cp.roll(zoomCh1, -1 * self.settings['demodCarrierIndex'], axis=0)
            zoomCh2 = cp.roll(zoomCh2, -1 * self.settings['demodCarrierIndex'], axis=0)

            # Second fft - ifft to get zoomed complex fringes
            zoomCh1 = self.settings['zoomFactor'] * self.settings['numSamples'] * cp.fft.ifft(zoomCh1, axis=0)
            zoomCh2 = self.settings['zoomFactor'] * self.settings['numSamples'] * cp.fft.ifft(zoomCh2, axis=0)
            # Interpolation for dechirping
            k1 = zoomCh1[self.chirp, :]
            k2 = zoomCh2[self.chirp, :]
            # Dispersion compensation
            k1 = k1 * self.dispersion[:, None]
            k2 = k2 * self.dispersion[:, None]

            # Log kspace if required
            if 'kspace' in self.settings['processState']:
                self.processedData['k1'][:, arrayRangePut] = cp.asnumpy(k1[:, 0:chunkSize])
                self.processedData['k2'][:, arrayRangePut] = cp.asnumpy(k2[:, 0:chunkSize])

            # multiply hanning window
            k1W = k1 * self.fringeWindow
            k2W = k2 * self.fringeWindow
            # Third fft
            tomch1 = cp.fft.fft(k1W, n=self.settings['numZOut'], axis=0) * 1e-6
            tomch2 = cp.fft.fft(k2W, n=self.settings['numZOut'], axis=0) * 1e-6
            k1W, k2W = None, None
            # Demodulation
            tomch1 = cp.roll(tomch1, +  self.settings['demodReverseIndex'], axis=0)
            tomch2 = cp.roll(tomch2, +  self.settings['demodReverseIndex'], axis=0)
            # Create Tomogram
            tomch1 = tomch1[0:self.settings['numZOut'], :]
            tomch2 = tomch2[0:self.settings['numZOut'], :]
            # Flip if required
            if self.settings['flipUpDown']:
                tomch1 = cp.flipud(tomch1)
                tomch2 = cp.flipud(tomch2)
            # Store as GPU array
            self.processedData['tomch1'][:, arrayRangePut] = tomch1[self.indexRange, 0:chunkSize]
            self.processedData['tomch2'][:, arrayRangePut] = tomch2[self.indexRange, 0:chunkSize]

            if 'ps' in self.settings['processState']:
                # GET SPECTRAL WINDOWS/BINS (CAN BE 1 BIN)
                if self.settings['spectralBinning']:
                    # multiply binned hanning windows
                    k1sbw = k1[:, :, None] * self.spectralFringeWindows
                    k2sbw = k2[:, :, None] * self.spectralFringeWindows
                    k1, k2 = None, None
                    # Third fft
                    tomch1SB = cp.fft.fft(k1sbw, n=self.settings['numZOut'], axis=0)
                    tomch2SB = cp.fft.fft(k2sbw, n=self.settings['numZOut'], axis=0)
                    k1sbw, k2sbw = None, None
                    # Demodulation
                    tomch1SB = cp.roll(tomch1SB, +  self.settings['demodReverseIndex'], axis=0) * 1e-6
                    tomch2SB = cp.roll(tomch2SB, +  self.settings['demodReverseIndex'], axis=0) * 1e-6
                    # Create Tomogram
                    tomch1SB = tomch1SB[0:self.settings['numZOut'], :, :]
                    tomch2SB = tomch2SB[0:self.settings['numZOut'], :, :]
                    # Flip if required
                    if self.settings['flipUpDown']:
                        tomch1SB = cp.flipud(tomch1SB)
                        tomch2SB = cp.flipud(tomch2SB)

                    tomch1PS, tomch2PS = tomch1SB, tomch2SB
                    tomch1SB, tomch2SB = None, None
                else:
                    tomch1PS, tomch2PS = tomch1, tomch2
                    tomch1, tomch2 = None, None

                # COMPUTE STOKES VECTORS UNCONCATENATED
                S0, S1, S2, S3 = stokes(tomch1PS[self.indexRange], tomch2PS[self.indexRange])
                tomch1PS, tomch2PS = None, None

                # CONCATENATE AND UNWEAVE INPUT POLIZARIZATION
                SV1, SV2 = unweaveInputPolarizations(S0, S1, S2, S3)
                S1, S2, S3 = None, None, None

                # Calculate downsampled stokes vectors and their padded positions
                # This is here to simplify the understanding of what's happening
                SV1, SV2, arrayRangeSplit = self.downsampleScanPattern(c, chunkSize, chunkSizePadded, SV1, SV2)

                # Convert to numpy array if limited by vram
                if self.settings['holdOnGPURam']:
                    self.processedData['sv1'][:, arrayRangeSplit + self.settings['chunkPadding'], :, :] = SV1
                    self.processedData['sv2'][:, arrayRangeSplit + self.settings['chunkPadding'], :, :] = SV2
                else:
                    self.processedData['sv1'][:, arrayRangeSplit + self.settings['chunkPadding'], :, :] = cp.asnumpy(SV1)
                    self.processedData['sv2'][:, arrayRangeSplit + self.settings['chunkPadding'], :, :] = cp.asnumpy(SV2)

                # Populate the edge positions of arrays for filtering during ps processing
                if self.settings['chunkPadding']>0:
                    if c == 0:
                        self.processedData['sv1'][:, 0:self.settings['chunkPadding'], :, :] = \
                            self.processedData['sv1'][:, self.settings['chunkPadding'], :, :][:, None, :, :]
                        self.processedData['sv2'][:, 0:self.settings['chunkPadding'], :, :] = \
                            self.processedData['sv2'][:, self.settings['chunkPadding'], :, :][:, None, :, :]
                    if c == numChunks - 1:
                        self.processedData['sv1'][:, -self.settings['chunkPadding']:, :, :] = \
                            self.processedData['sv1'][:, -self.settings['chunkPadding']-1, :, :][:, None, :,:]
                        self.processedData['sv2'][:, -self.settings['chunkPadding']:, :, :] = \
                            self.processedData['sv2'][:, -self.settings['chunkPadding']-1, :, :][:, None, :,:]

        self.ch1, self.ch2 = None, None
        SV1, SV2 = None, None
        tomch1, tomch2, k1, k2, k1sbw, k2sbw = None, None, None, None, None, None
        dataChunkX, dataChunkY, zoomCh1, zoomCh2, tomch1SB, tomch2SB = None, None, None, None, None, None

    def heterodyne1channel(self):
        """
        Compute an optical coherence tomogram using heterodyne 2-channel polarization diverse detection

        Note:
            If PS reconstruction enabled, both am unbinned and a binned complex tomogram  will be produced.
            Reference(s):
            Tomogram reconstruction:
            [1] Yun, S., Tearney, G., de Boer, J., Iftimia, N., & Bouma, B. (2003).
                High-speed optical frequency-domain imaging. Optics Express, 11(22), 2953.
                https://doi.org/10.1364/oe.11.002953
            [2] Motaghian Nezam, S. M. R., Vakoc, B. J., Desjardins, A. E., Tearney, G. J., & Bouma, B. E. (2007).
                Increased ranging depth in optical frequency domain imaging by frequency encoding. Optics Letters,
                32(19), 2768.
                https://doi.org/10.1364/ol.32.002768
            Stokes Vector calculation:
            [1] Park BH, Pierce MC, Cense B, de Boer JF.
                Real-time multi-functional optical coherence tomography.
                Opt Express. 2003;11(7):782–793
        Args:
            param1 (obj): data
        Output:
            self.processedData['tomch1'], self.processedData['tomch2'], self.SV1, self,SV2

        """

        # try:
        # Initialize on GPU Memory and remove bg1
        self.processedData['tomch1'][:] = 0


        self.ch1[:, 0::2] = self.ch1[:, 0::2] - self.bgv1[:, None]
        self.ch1[:, 1::2] = self.ch1[:, 1::2] - self.bgh1[:, None]

        chunkSize = int(self.ch1.shape[1] / self.numChunks)
        numChunks = int(cp.floor(self.ch1.shape[1] / chunkSize))
        logging.info('Chunk Size : {}'.format(chunkSize))
        logging.info('Number of chunks: {}'.format(numChunks))

        # Grab a little extra to make sure we complete the downsample properly
        chunkSizePadded = closestDividableNumber(chunkSize, self.settings['imgDepth'] * 2)
        chunkSizeDifference = chunkSizePadded - chunkSize
        zoomCh1 = cp.zeros((self.settings['numSamples'] * self.settings['zoomFactor'], chunkSizePadded),
                           dtype='complex64')
        for c in range(numChunks):
            arrayRangeGrabbed = np.arange(chunkSize * c, (chunkSize) * (c + 1) + chunkSizeDifference, 1)
            arrayRangePut = np.arange(chunkSize * c, (chunkSize) * (c + 1), 1)
            # Apply first fft
            if c == numChunks - 1:
                dataChunkX = cp.zeros((self.ch1.shape[0], chunkSizePadded), dtype='complex64')
                dataChunkX[:, 0:len(arrayRangePut)] = cp.fft.fft(self.ch1[:, arrayRangePut], axis=0)
            else:
                dataChunkX = cp.fft.fft(self.ch1[:, arrayRangeGrabbed], axis=0)
            zoomCh1[:] = 0
            # Take real data
            zoomCh1[0:self.fourierLength, :] = dataChunkX[0:self.fourierLength, :]
            dataChunkX = None
            # Demodulation
            zoomCh1 = cp.roll(zoomCh1, -1 * self.settings['demodCarrierIndex'], axis=0)
            # Second fft - ifft to get zoomed complex fringes
            zoomCh1 = self.settings['zoomFactor'] * self.settings['numSamples'] * cp.fft.ifft(zoomCh1, axis=0)
            # Interpolation for dechirping
            k1 = zoomCh1[self.chirp, :]
            # Dispersion compensation
            k1 = k1 * self.dispersion[:, None]
            if 'kspace' in self.settings['processState']:
                self.processedData['k1'][:, arrayRangePut] = cp.asnumpy(k1[:, 0:chunkSize])
            # multiply hanning window
            k1W = k1 * self.fringeWindow
            # Third fft
            tomch1 = cp.fft.fft(k1W, n=self.settings['numZOut'], axis=0) * 1e-6
            k1W = None
            # Demodulation
            tomch1 = cp.roll(tomch1, +  self.settings['demodReverseIndex'], axis=0)
            # Create Tomogram
            tomch1 = tomch1[0:self.settings['numZOut'], :]
            # Flip if required
            if self.settings['flipUpDown']:
                tomch1 = cp.flipud(tomch1)
            # Store as GPU array
            self.processedData['tomch1'][:, arrayRangePut] = tomch1[self.indexRange, 0:chunkSize]

    def minimal2channel(self):
        """Single fft from K to Z space (no correction, zoom, or modulation) """
        self.processedData['tomch1'][:] = 0
        self.processedData['tomch2'][:] = 0

        # self.ch1[:, 0::2] = self.ch1[:, 0::2] - self.bgv1[:, None]
        # self.ch1[:, 1::2] = self.ch1[:, 1::2] - self.bgh1[:, None]
        # self.ch2[:, 0::2] = self.ch2[:, 0::2] - self.bgv2[:, None]
        # self.ch2[:, 1::2] = self.ch2[:, 1::2] - self.bgh2[:, None]

        tomch1 = cp.fft.fft(self.ch1, n=self.fourierLength, axis=0)
        tomch2 = cp.fft.fft(self.ch2, n=self.fourierLength, axis=0)

        if self.settings['flipUpDown']:
            tomch1 = cp.flipud(tomch1)
            tomch2 = cp.flipud(tomch2)

        self.processedData['tomch1'] = tomch1[self.indexRange, :]
        self.processedData['tomch2'] = tomch2[self.indexRange, :]

        if 'kspace' in self.settings['processState']:
            self.processedData['k1'] = cp.asnumpy(self.ch1)
            self.processedData['k2'] = cp.asnumpy(self.ch2)

        # if 'ps' in self.settings['processState']:
        #     self.processedData['sv1'] = tomch1
        #     self.processedData['sv2'] = tomch2

        tomch1, tomch2 = None, None

    def minimal1channel(self):
        """Single fft from K to Z space (no correction, zoom, or modulation) """
        self.processedData['tomch1'][:] = 0

        # self.ch1[:, 0::2] = self.ch1[:, 0::2] - self.bgv1[:, None]
        # self.ch1[:, 1::2] = self.ch1[:, 1::2] - self.bgh1[:, None]

        tomch1 = cp.fft.fft(self.ch1, n=self.fourierLength, axis=0)

        if self.settings['flipUpDown']:
            tomch1 = cp.flipud(tomch1)

        self.processedData['tomch1'] = tomch1[self.indexRange, :]

        if 'kspace' in self.settings['processState']:
            self.processedData['k1'] = cp.asnumpy(self.ch1)

        # if 'ps' in self.settings['processState']:
        #     self.processedData['sv1'] = tomch1

        tomch1 = None

    def downsampleScanPattern(self, c, chunkSize, chunkSizePadded, SV1, SV2):
        """
        Downsamples stokes vectors to a normal frame size.
        Specifically for bi-seg scan patterns to increase processing speed
        """
        # TAKE MEAN OVER IMAGE DEPTH
        if self.settings['imgDepth'] > 1:
            if self.settings['fastProcessing']:
                SV1 = cp.mean(SV1.reshape(len(self.indexRange),
                                          int(chunkSizePadded / 2 / self.settings['imgDepth']),
                                          self.settings['imgDepth'], 4,
                                          self.settings['numSpectralWindows']), axis=2)
                SV2 = cp.mean(SV2.reshape(len(self.indexRange),
                                          int(chunkSizePadded / 2 / self.settings['imgDepth']),
                                          self.settings['imgDepth'], 4,
                                          self.settings['numSpectralWindows']), axis=2)

                # Calculate "downsampled" position
                d = np.int(chunkSizePadded / self.settings['imgDepth'] / 2) - \
                    np.int(chunkSize / self.settings['imgDepth'] / 2)

                arrayRangeSplit = np.arange(np.int(chunkSize / 2 / self.settings['imgDepth']) * c,
                                            np.int(chunkSize / 2 / self.settings['imgDepth']) * (c + 1), 1)
                arrayRangeCurr = np.arange(0, chunkSizePadded / 2 / self.settings['imgDepth'] - d, 1, dtype='int')

            # KEEP RESOLUTION USING BISEG SCAN PATTERN OVERSAMPLING
            else:
                SV1 = SV1.reshape(len(self.indexRange),
                                  int(chunkSizePadded / 2 / self.settings['imgDepth']),
                                  self.settings['imgDepth'], 4, self.settings['numSpectralWindows'])

                SV1Recon = cp.zeros((SV1.shape[0], SV1.shape[1] * 2, SV1.shape[3], SV1.shape[4]))
                SV1Recon[:, 0::2, :, :] = cp.mean(SV1[:, :, 0:int(self.settings['imgDepth'] / 2), :, :],axis=2)
                SV1Recon[:, 1::2, :, :] = cp.mean(SV1[:, :, int(-self.settings['imgDepth'] / 2):, :, :],axis=2)

                SV2 = SV2.reshape(len(self.indexRange),
                                  int(chunkSizePadded / 2 / self.settings['imgDepth']),
                                  self.settings['imgDepth'], 4, self.settings['numSpectralWindows'])

                SV2Recon = cp.zeros((SV2.shape[0], SV2.shape[1] * 2, SV2.shape[3], SV2.shape[4]))
                SV2Recon[:, 0::2, :, :] = cp.mean(SV2[:, :, 0:int(self.settings['imgDepth'] / 2), :, :], axis=2)
                SV2Recon[:, 1::2, :, :] = cp.mean(SV2[:, :, int(self.settings['imgDepth'] / 2):, :, :], axis=2)

                # Calculate "downsampled" position
                d = np.int(chunkSizePadded / self.settings['imgDepth']) - np.int(chunkSize / self.settings['imgDepth'])
                arrayRangeSplit = np.arange(np.int(chunkSize / self.settings['imgDepth']) * c,
                                            np.int(chunkSize / self.settings['imgDepth']) * (c + 1), 1, dtype='int')
                arrayRangeCurr = np.arange(0, chunkSizePadded / self.settings['imgDepth'] - d, 1, dtype='int')
                SV1 = SV1Recon
                SV2 = SV2Recon
        else:
            d = np.int(chunkSizePadded / 2) - np.int(chunkSize / 2)
            arrayRangeSplit = np.arange(np.int(chunkSize / 2 / self.settings['imgDepth']) * c,
                                        np.int(chunkSize / 2 / self.settings['imgDepth']) * (c + 1), 1, dtype='int')
            arrayRangeCurr = np.arange(0, chunkSizePadded / 2 / self.settings['imgDepth'] - d, 1, dtype='int')
            # Make sure dataset is right size(X,X,X,1)
            if not self.settings['spectralBinning']:
                SV1 = SV1[:, :, :, None]
                SV2 = SV2[:, :, :, None]

        return SV1[:, arrayRangeCurr, :, :], SV2[:, arrayRangeCurr, :, :], arrayRangeSplit



