from itertools import filterfalse
from oct.utils import *
from oct.utils.angioUtils import *
from ..load.metadata import Metadata
import logging

cp, np, convolve, gpuAvailable, freeMemory, e  = checkForCupy()

class Angiography:
    """ Angiography contrast OCT reconstruction"""
    def __init__(self, mode='cdv'):

        acceptedModes = 'cdv+cdv_nb'
        if mode in acceptedModes:
            self.mode = mode
        else:
            self.mode = 'cdv'

        self.filter = None
        self.initialized = False
        self.chCount = 0

        self.tomch1 = None
        self.tomch2 = None

        self.processedData = {
            'angio': None,
            'weight': None
        }

        meta = Metadata()
        self.settings = meta.angioSettings


    def initialize(self, data=None, settings=None):
        """
        Initialize the angiography reconstruction with desired settings
        Args:
            data (object) : Support and raw data holder
            settings (dict) : Manual input settings holder
            filterSize (tuple) : filter X & Z sizes
        """

        self.setSettings(data=data, settings=settings)

        if self.filter is None:
            d1 = cp.hanning(self.settings['xFilter'])
            d2 = cp.hanning(self.settings['zFilter'])
            self.filter = cp.sqrt(cp.outer(d1, d2))
            self.filter = self.filter / cp.sum(self.filter)

        self.initialized = True

        logging.info('====================================================')
        logging.info('Angio settings initialized:')
        for i in self.settings.keys():
            logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.settings[i]))

    def setSettings(self, data=None, settings=None):
        """
        Sets the reconstruction settings for structural contrast
        Notes:

        Args:
            data (object) : An object containing all the preloaded data from a directory
            settings (dict): The required/edittable settings to process an angio contrast frame
        Yields:
            self.settings
        """
        if data:
            for key, val in data.angioSettings.items():
                self.settings[key] = data.angioSettings[key]

        elif settings:
            for key, val in settings.items():
                self.settings[key] = settings[key]

    def requires(self):
        """Prints out the required/optional variables to perform reconstruction"""
        print('Required:')
        print("tomch1=tomch1")
        print("tomch2=tomch2")
        print("data.angioSettings['xFilter'] (for bi-seg scans) ")
        print("data.angioSettings['zFilter'] (for bi-seg scans) ")
        print("data.angioSettings['imgWidthAng'] (for bi-seg scans) ")
        print("data.angioSettings['imgDepthAng'] (for bi-seg scans) ")
        print("data.angioSettings['AlinesToProcAngioLinesA'] (for bi-seg scans) ")
        print("data.angioSettings['AlinesToProcAngioLinesB'] (for bi-seg scans) ")

        print("\n")
        print('Optional:')
        print("data.angioSettings['invertGray']")
        print("data.angioSettings['contrastLowHigh'] ( [min, max] )")

        print("\n")
        print('For reference, the whole settings dict and its defaults are:')
        for key, value in self.settings.items():
            print("self.settings['", key, "'] : ", value)

    def manageChannels(self):
        """ Handle case where only channel 2's are passed """
        if not (self.tomch2 is None) and (self.tomch1 is None):
            self.tomch1 = self.tomch2
            self.tomch1 = None

    def reconstruct(self, tomch1=None, tomch2=None, data=None, settings=None):
        """
        Reconstruct an angiography contrast frame

        Notes:
        Args:
            tomch1 (array) : Reconstructed tomogram from channel 1
            tomch1 (array) : Reconstructed tomogram from channel 2
            data (object) : An object containing all the preloaded data from a directory for post processing
            settings (dict): The required/edittable settings to process an angio contrast frame
        Returns:
            processeData (dict) : A dictionary containing an angio and average intensity contrast frame labelled
                'angio' and 'weight', respectively
        """

        if not self.initialized:
            if data:
                if data:
                    self.tomch1 = cp.asarray(data.processedData['tomch1'])
                    self.tomch2 = cp.asarray(data.processedData['tomch2'])
            else:
                if not (tomch1 is None):
                    self.chCount = self.chCount + 1
                    self.tomch1 = cp.asarray(tomch1)
                if not (tomch2 is None):
                    self.chCount = self.chCount + 1
                    self.tomch2 = cp.asarray(tomch2)
            self.manageChannels()
            self.initialize(data=data, settings=settings)

        if self.mode == 'cdv':
            self.cdv()
        if self.mode == 'cdvnoisebias':
            self.cdv_nb()

        self.fixImageWidth()

        return self.processedData

    def cdv(self):
        """
        Compute angiography frames using the CDV algorithm
        Note:
            More specifically, this function calculates the complex differential variance between alines
            taken at the same position. This is therefore intensity based.
            Reference:
            [1] Nam, A. S., Chico-Calero, I., & Vakoc, B. J. (2014).
                Complex differential variance algorithm for optical coherence tomography angiography.
                Biomedical Optics Express, 5(11), 3822.
                https://doi.org/10.1364/boe.5.003822
        Args:

        Output:
            self.processedData['angio'] (np.array): Reconstructed angiography frame
            self.processedData['weight'] (np.array): Reconstructed weight frame
        """

        tomCh11 = cp.squeeze(self.tomch1[:, self.settings['AlinesToProcAngioLinesA']])
        tomCh12 = cp.squeeze(self.tomch1[:, self.settings['AlinesToProcAngioLinesB']])
        tomCh21 = cp.squeeze(self.tomch2[:, self.settings['AlinesToProcAngioLinesA']])
        tomCh22 = cp.squeeze(self.tomch2[:, self.settings['AlinesToProcAngioLinesB']])

        absZsumCh1 = tomCh11 * cp.conj(tomCh12)
        absZsumCh2 = tomCh21 * cp.conj(tomCh22)

        # Intensity average frames for projection weight masking
        intensityAvgCh1 = 0.5 * (cp.abs(tomCh11) ** 2 + cp.abs(tomCh12) ** 2)
        intensityAvgCh2 = 0.5 * (cp.abs(tomCh21) ** 2 + cp.abs(tomCh22) ** 2)

        ch11, ch12, ch21, ch22 = None, None, None, None

        # Convolutional filter Zsum frames (does not support complex, so is split)
        absZsumCh1Real = convolve(cp.real(absZsumCh1), self.filter, mode='constant').astype('float32')
        absZsumCh1Imag = convolve(cp.imag(absZsumCh1), self.filter, mode='constant').astype('float32')
        absZsumCh1 = absZsumCh1Real + absZsumCh1Imag * 1j
        absZsumCh1Real, absZsumCh1Imag = None, None

        absZsumCh2Real = convolve(cp.real(absZsumCh2), self.filter, mode='constant').astype('float32')
        absZsumCh2Imag = convolve(cp.imag(absZsumCh2), self.filter, mode='constant').astype('float32')
        absZsumCh2 = absZsumCh2Real + absZsumCh2Imag * 1j
        absZsumCh2Real, absZsumCh2Imag = None, None

        # Convolutional filter average intensity frames
        intensityAvgCh1 = convolve(intensityAvgCh1, self.filter, mode='constant')
        intensityAvgCh2 = convolve(intensityAvgCh2, self.filter, mode='constant')

        # Normalize convolution scaling
        absZsumCh1 = cp.absolute(absZsumCh1) / cp.sum(self.filter)
        absZsumCh2 = cp.absolute(absZsumCh2) / cp.sum(self.filter)
        intensityAvgCh1 = intensityAvgCh1 / cp.sum(self.filter)
        intensityAvgCh2 = intensityAvgCh2 / cp.sum(self.filter)

        intensityMask = (intensityAvgCh1 >= intensityAvgCh2)

        # Combine channels based on weight
        intensityAvg = intensityAvgCh1 * intensityMask + intensityAvgCh2 * np.invert(intensityMask)
        complexZsum = absZsumCh1 * intensityMask + absZsumCh2 * np.invert(intensityMask)
        intensityMask, absZsumCh1, absZsumCh2, intensityAvgCh1, intensityAvgCh2 = None, None, None, None, None

        # Reshape based on angio/biseg scan pattern
        complexZsum = complexZsum.reshape([complexZsum.shape[0],
                                           int(complexZsum.shape[1]/self.settings['imgDepthAng']),
                                           self.settings['imgDepthAng']])
        intensityAvg = intensityAvg.reshape([intensityAvg.shape[0],
                                             int(intensityAvg.shape[1] / self.settings['imgDepthAng']),
                                             self.settings['imgDepthAng']])

        num = cp.sum(complexZsum, axis=2)
        weight = cp.sum(intensityAvg, axis=2)
        complexZsum, intensityAvg = None, None

        angio = 1 - num / weight
        num, den = None, None

        # Scale and clip images
        self.processedData['angio'] = self.formatAngioOut(angio)
        self.processedData['weight'] = self.formatWeightOut(weight)

        angio, weight = None, None

    def cdv_nb(self):
        """
        Compute angiography frames using the CDV algorithm with noise-bias correction
        Note:
            This is an upgrade to the previous CDV algorithm that improves performance using a noise-bias correction.
            Reference:
            Braaf, B., Donner, S., Nam, A. S., Bouma, B. E., & Vakoc, B. J. (2018).
            Complex differential variance angiography with noise-bias correction for optical coherence tomography of the retina.
            Biomedical Optics Express, 9(2), 486.
            https://doi.org/10.1364/boe.9.000486
        Args:

        Output:
            self.processedData['angio'] (np.array): Reconstructed angiography frame
            self.processedData['weight'] (np.array): Reconstructed weight frame
        """
        ## TODO ##
        pass


    def mrcdv_on_segment(self, tomch1, tomch2, mrcdv):
        tomCh11 = cp.squeeze(tomch1[:, mrcdv.segment['AlinesToProcAngioLinesA']])
        tomCh12 = cp.squeeze(tomch1[:, mrcdv.segment['AlinesToProcAngioLinesB']])
        tomCh21 = cp.squeeze(tomch2[:, mrcdv.segment['AlinesToProcAngioLinesA']])
        tomCh22 = cp.squeeze(tomch2[:, mrcdv.segment['AlinesToProcAngioLinesB']])

        absZsumCh1 = tomCh11 * cp.conj(tomCh12)
        absZsumCh2 = tomCh21 * cp.conj(tomCh22)

        # Intensity average frames for projection weight masking
        intensityAvgCh1 = 0.5 * (cp.abs(tomCh11) ** 2 + cp.abs(tomCh12) ** 2)
        intensityAvgCh2 = 0.5 * (cp.abs(tomCh21) ** 2 + cp.abs(tomCh22) ** 2)

        ch11, ch12, ch21, ch22 = None, None, None, None

        # Convolutional filter Zsum frames (does not support complex, so is split)
        absZsumCh1Real = convolve(cp.real(absZsumCh1), self.filter, mode='constant').astype('float32')
        absZsumCh1Imag = convolve(cp.imag(absZsumCh1), self.filter, mode='constant').astype('float32')
        absZsumCh1 = absZsumCh1Real + absZsumCh1Imag * 1j
        absZsumCh1Real, absZsumCh1Imag = None, None

        absZsumCh2Real = convolve(cp.real(absZsumCh2), self.filter, mode='constant').astype('float32')
        absZsumCh2Imag = convolve(cp.imag(absZsumCh2), self.filter, mode='constant').astype('float32')
        absZsumCh2 = absZsumCh2Real + absZsumCh2Imag * 1j
        absZsumCh2Real, absZsumCh2Imag = None, None

        # Convolutional filter average intensity frames
        intensityAvgCh1 = convolve(intensityAvgCh1, self.filter, mode='constant')
        intensityAvgCh2 = convolve(intensityAvgCh2, self.filter, mode='constant')

        # Normalize convolution scaling
        absZsumCh1 = cp.absolute(absZsumCh1) / cp.sum(self.filter)
        absZsumCh2 = cp.absolute(absZsumCh2) / cp.sum(self.filter)
        intensityAvgCh1 = intensityAvgCh1 / cp.sum(self.filter)
        intensityAvgCh2 = intensityAvgCh2 / cp.sum(self.filter)

        intensityMask = (intensityAvgCh1 >= intensityAvgCh2)

        # Combine channels based on weight
        intensityAvg = intensityAvgCh1 * intensityMask + intensityAvgCh2 * np.invert(intensityMask)
        complexZsum = absZsumCh1 * intensityMask + absZsumCh2 * np.invert(intensityMask)
        intensityMask, absZsumCh1, absZsumCh2, intensityAvgCh1, intensityAvgCh2 = None, None, None, None, None

        # Reshape based on angio/biseg scan pattern
        complexZsum = complexZsum.reshape([complexZsum.shape[0],
                                           cp.int_(complexZsum.shape[1]/mrcdv.segment['imgDepthAng']),
                                           mrcdv.segment['imgDepthAng']])
        intensityAvg = intensityAvg.reshape([intensityAvg.shape[0],
                                             cp.int_(intensityAvg.shape[1] / mrcdv.segment['imgDepthAng']),
                                             mrcdv.segment['imgDepthAng']])

        return (complexZsum, intensityAvg)


    def mrcdv_on_frame(self, mrcdv, b_monitor=False):

        f_angio = cp.zeros((self.tomch1.shape[0], self.settings['imgWidthAng']))
        f_weight = cp.zeros((self.tomch1.shape[0], self.settings['imgWidthAng']))
        M = mrcdv.settings['M'] 
        num_segments = mrcdv.settings['num_segments']
        num_xlocs = int(self.settings['imgWidthAng']/num_segments)
        num_pairs = mrcdv.settings['num_pairs']
        for qq_segment in range(num_segments):
            slice_xloc_angio = slice(qq_segment*num_xlocs, (qq_segment+1)*num_xlocs)
            slice_xloc_tomo = slice(qq_segment*num_xlocs*M, (qq_segment+1)*num_xlocs*M)


            complexZsum0, intensityAvg0 = self.mrcdv_on_segment(self.tomch1[:,slice_xloc_tomo], self.tomch2[:, slice_xloc_tomo], mrcdv)
            temp = 1 - complexZsum0/intensityAvg0

            black_sum = cp.sum(cp.logical_and(temp<mrcdv.settings['c_cdv'], intensityAvg0>mrcdv.settings['c_int']), axis=(0,1))
            ind_maxangio = cp.argsort(black_sum)
            if b_monitor:
                print("ind_maxangio:", ind_maxangio)

            num = cp.sum(complexZsum0[:,:,ind_maxangio[-num_pairs:]], axis=2)
            den = cp.sum(intensityAvg0[:,:,ind_maxangio[-num_pairs:]], axis=2)

            f_angio[:,slice_xloc_angio] = 1 - num/den
            f_weight[:,slice_xloc_angio] = den
            num, den = None, None

        # Scale and clip images
        self.processedData['angio'] = self.formatAngioOut(f_angio)
        self.processedData['weight'] = self.formatWeightOut(f_weight)
        f_angio, f_weight = None, None





        


    def formatWeightOut(self, weight):
        """ Format the output weight array to Uint8, according to contrast settings

        Notes:

        Args:
            weight (array) : Un-formatted float intensity array
        Returns:
            array : Uint8 (0-255) formatted weight array within contrast range
        """
        weight = 10 * cp.log10(weight / self.settings['imgDepthAng'])
        weight = (weight - self.settings['contrastLowHigh'][0]) / \
                 (self.settings['contrastLowHigh'][1] - self.settings['contrastLowHigh'][0])
        weight = (cp.clip(weight, a_min=0, a_max=1) * 255)  # clip and scale for uint8
        return cp.asnumpy(weight.astype('uint8'))

    def formatAngioOut(self, angio):
        """ Format the output angio array to Uint8, according to contrast settings

        Notes:

        Args:
            angio (array) : Un-formatted float intensity array
        Returns:
            array : Uint8 (0-255) formatted angio array
        """
        angio = cp.real(cp.sqrt(angio))  # Get real values to avoid the digital filtering's edge effect.
        angio = cp.clip(angio, a_min=0, a_max=1)
        if self.settings['invertGray']:
            angio = angio*-1+1
        angio = cp.asnumpy((angio * 255).astype('uint8')) # clip and scale for uint8
        return angio

    def fixImageWidth(self):
        """
        If ramp scan patter angio is being used, we need to add a column to the data to make it equal to the
        other datasets.
        """
        if self.settings['imgDepth']==1:
            for key, val in self.processedData.items():
                self.processedData[key] = np.c_[self.processedData[key],
                                                1e-15+np.zeros(self.processedData[key].shape[0])]