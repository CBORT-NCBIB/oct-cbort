from oct.utils.psUtils import *
from ..load.metadata import Metadata
import math
import logging

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()


class Polarization:
    """ Polarization contrast OCT reconstruction"""

    def __init__(self, mode='sym'):

        acceptedModes = 'rt+classic+sb+sym+mm'
        if mode in acceptedModes:
            self.mode = mode
        else:
            self.mode = 'sym'

        self.filter = None
        self.initialized = False
        self.chCount = 0

        self.sv1 = None
        self.sv2 = None

        self.processedData = {
            'dop': None,
            'ret': None,
            'oa': None
        }

        self.corrections = {
            'symmetry': None,
            'bins': None,
        }

        meta = Metadata()
        self.settings = meta.psSettings

    def initialize(self, data=None, settings=None):
        """
        Initialize the PS portion of the reconstruction with desired settings
        Note:
            This needs to be cleaned up once the dllsettings.ini file has been updated
            PS reconstruction is not something that is always required, so it is kept seperate from regular normalization.
        Args:
            data (object) : Support and raw data holder
            binFract (int) : fraction of spectral fringe in each bin
            xFilter (int) : filter size applied across alines at beginning of ps computation
            zFilter (int) : filter size applied along alines at beginning of ps computation
            oopFilter (int) : filter size applied across frames at end of compute tomogram for SV1 + SV2
            maxRet (int) : normalization factor for the maximum retardance
            zOffset (int) : offset in Z over which to compute retardance, ie. the "self.settings['zOffset']"
            zResolution (int) : resolution of system in Z
            dopThresh (int) : the lower bound of the DOP threshold (upper bound is always 1)
            overwriteSettings (int) : allows the overwriting of file initialized settings
        """

        self.setSettings(data=data, settings=settings)
        
        if self.filter is None:
            d1 = cp.hanning(self.settings['zFilter'])
            d2 = cp.hanning(self.settings['xFilter'])
            self.filter = cp.sqrt(cp.outer(d1, d2))
            self.filter = self.filter / cp.sum(self.filter)
        self.initialized = True

        logging.info('====================================================')
        logging.info('PS settings initialized:')
        for i in self.settings.keys():
            logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.settings[i]))

    def setSettings(self, data=None, settings=None):
        """
        Sets the reconstruction settings for ps contrast
        Notes:

        Args:
            data (object) : An object containing all the preloaded data from a directory
            settings (dict): The required/edittable settings to process an angio contrast frame
        Yields:
            self.settings
        """
        if data:
            for key, val in data.psSettings.items():
                self.settings[key] = data.psSettings[key]

        elif settings:
            for key, val in settings.items():
                self.settings[key] = settings[key]

    def setOutputName(self):
        """Creates output name string, given input settings"""
        self.psString = '[ps.[{} {} {} {}]]'.format(self.settings['maxRet'],
                                                    self.settings['numSpectralWindows'],
                                                    self.settings['zOffset'],
                                                    self.settings['xFilter'])

    def requires(self):
        """Prints out the required/optional variables to perform reconstruction"""
        print('Required:')
        print("sv1=sv1")
        print("\n")
        print('Optional:')
        print("\n")
        print("sv2=sv2")
        print("\n")
        print('OR:')
        print("data=data")
        print("\n")
        print('For settings changes:\n'
              'settings=settings:')
        print("\n")
        print('For reference, the whole settings dict and its defaults are:')
        for key, value in self.settings.items():
            print("self.settings['", key, "'] : ", value)

    def manageChannels(self):
        """ Handle case where only channel 2's are passed """
        if not (self.sv1 is None) and (self.sv2 is None):
            self.sv1 = self.sv2
            self.sv2 = None

    def reconstruct(self, sv1=None, sv2=None, data=None, settings=None, corrections=None):
        """
        Reconstruct ps contrast frames

        Notes:
        Args:
            sv1 (array) : Reconstructed stokes vectors from input state 1
            sv2 (array) : Reconstructed stokes vectors from input state 2
            data (object) : An object containing all the preloaded data from a directory for post processing
            settings (dict): The required/edittable settings to process an angio contrast frame
        Returns:
             processeData (dict) : A dictionary containing the following contrasts:
                                    Degree of polarization : 'dop'
                                    Retardance : 'ret'
                                    Apparent Optic axis : 'oa'
        """

        if not self.initialized:
            if corrections:
                self.corrections = corrections
            if data:
                if data:
                    self.sv1 = cp.asarray(data.processedData['sv1'])
                    self.sv2 = cp.asarray(data.processedData['sv2'])
                    self.corrections = data.psCorrections
            else:
                if not (sv1 is None):
                    self.chCount = self.chCount + 1
                    self.sv1 = cp.asarray(sv1)
                if not (sv2 is None):
                    self.sv2 = self.chCount + 1
                    self.sv2 = cp.asarray(sv2)
            self.manageChannels()
            self.initialize(data=data, settings=settings)
            if data is None and settings is None:
                self.settings['imgWidth']= self.sv1.shape[1]
        else:
            if data:
                self.sv1 = cp.asarray(data.processedData['sv1'])
                self.sv2 = cp.asarray(data.processedData['sv2'])
            else:
                if not (sv1 is None):
                    self.sv1 = cp.asarray(sv1)
                if not (sv2 is None):
                    self.sv2 = cp.asarray(sv2)

        if self.mode == 'rt':
            oa, dop = self.realTime()
        elif self.mode == 'classic':
            oa, dop = self.classic()
        elif self.mode == 'sb':
            oa, dop = self.spectralBinning()
        elif self.mode == 'sym': # Default
            oa, dop = self.symmetric()
        elif self.mode == 'mm':
            oa, dop = self.measurementMat()


        self.processedData['ret'], \
        self.processedData['dop'], \
        self.processedData['oa'], \
        self.processedData['theta'] = self.formatOut(oa, dop)

        # for key, val in self.processedData.items():
        #     self.processedData[key] = self.clearPadding(self.processedData[key])

        return self.processedData

    def realTime(self):
        """
        Compute retardance, degree of polarization, and optic axis frames from PS sensitive datasets

        Note:
            Reference:
            [1]


            Spectral binning is not used here.
        Args:

        Output:
            self.processedData['ret'] (np.array) : retardance (100 deg/um )
            self.processedData['dop'] (np.array) : degree of polarization
            self.processedData['oa'] (np.array) : apparent optic axis
        """
        SV1 = cp.asarray(self.sv1[:, :, 1:4, :])
        SV2 = cp.asarray(self.sv2[:, :, 1:4, :])

        S1, S2, S3 = None, None, None

        # i^2 = q^2+u^2+v^2 (eq.3.36 Theocaris)>>
        I1 = cp.sqrt((SV1 * SV1).sum(axis=2))
        I2 = cp.sqrt((SV2 * SV2).sum(axis=2))

        # Filter
        SV1 = convolve(SV1, self.filter[:, :, None, None], mode='constant')
        SV2 = convolve(SV2, self.filter[:, :, None, None], mode='constant')
        I1 = convolve(I1, self.filter[:, :, None], mode='constant')
        I2 = convolve(I2, self.filter[:, :, None], mode='constant')

        If = cp.mean(I1 * I1 + I2 * I2, axis=-1)

        # Euclidian length of Q,U,V <<after averaging, so different from I1,I2>>
        I1 = (SV1 * SV1).sum(axis=2)
        I2 = (SV2 * SV2).sum(axis=2)

        dop = cp.sqrt(cp.mean(I1 + I2, axis=-1) / If)

        SV1 = SV1 / cp.sqrt(I1)[:, :, None, :]
        SV2 = SV2 / cp.sqrt(I2)[:, :, None, :]
        SV1 = SV1 + SV2
        SV2 = SV1 - 2 * SV2
        nna = cp.sqrt((SV1 * SV1).sum(axis=2))
        nnb = cp.sqrt((SV2 * SV2).sum(axis=2))
        SV1 = SV1 / nna[:, :, None]
        SV2 = SV2 / nnb[:, :, None]

        I1, I2, If, nna, nnb = None, None, None, None, None

        SV1minus = cp.roll(SV1, -self.settings['zOffset'], axis=0)
        SV1 = cp.roll(SV1, self.settings['zOffset'], axis=0)
        SV2minus = cp.roll(SV2, -self.settings['zOffset'], axis=0)
        SV2 = cp.roll(SV2, self.settings['zOffset'], axis=0)

        # Normalized Cross Product
        oa = cp.cross(SV1 - SV1minus, SV2 - SV2minus, axis=2)
        den = cp.sqrt((oa * oa).sum(axis=2))
        den = cp.clip(den, a_min=1e-15, a_max=None)
        oa = oa / den[:, :, None]

        den = None

        # Calculate retsinW
        temp = (SV1minus * oa).sum(axis=2) ** 2
        temp = cp.clip(temp,a_min=1e-15,a_max=1-.0000001)
        temp2 = ((SV1minus * SV1).sum(axis=2) - temp) / (1-temp)
        retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']

        temp, temp2 = None, None

        pm = cp.sign((1 - (SV1minus * SV1).sum(axis=2)) * ((SV1minus - SV1) * (SV2minus + SV2)).sum(axis=2))
        SV2, SV1, SV1minus, SV2minus = None, None, None, None

        oa = oa * cp.expand_dims(pm, axis=2) * cp.expand_dims(retSinW, axis=2)
        retSinW = None

        oa = cp.mean(oa, axis=-1)

        return oa, dop

    def classic(self):
        """
        Compute retardance, degree of polarization, and optic axis frames from PS sensitive datasets

        Note:
            Reference:
            [1]


            Spectral binning is not used here.
        Args:

        Output:
            self.processedData['ret'] (np.array) : retardance (100 deg/um )
            self.processedData['dop'] (np.array) : degree of polarization
            self.processedData['oa'] (np.array) : apparent optic axis
        """
        SV1 = cp.asarray(self.sv1[:, :, 1:4, :])
        SV2 = cp.asarray(self.sv2[:, :, 1:4, :])

        S1, S2, S3 = None, None, None

        # i^2 = q^2+u^2+v^2 (eq.3.36 Theocaris)>>
        I1 = cp.sqrt((SV1 * SV1).sum(axis=2))
        I2 = cp.sqrt((SV2 * SV2).sum(axis=2))

        # Filter
        SV1 = convolve(SV1, self.filter[:, :, None, None], mode='constant')
        SV2 = convolve(SV2, self.filter[:, :, None, None], mode='constant')
        I1 = convolve(I1, self.filter[:, :, None], mode='constant')
        I2 = convolve(I2, self.filter[:, :, None], mode='constant')

        If = cp.mean(I1 * I1 + I2 * I2, axis=-1)

        # Euclidian length of Q,U,V <<after averaging, so different from I1,I2>>
        I1 = (SV1 * SV1).sum(axis=2)
        I2 = (SV2 * SV2).sum(axis=2)

        dop = cp.sqrt(cp.mean(I1 + I2, axis=-1) / If)

        SV1 = SV1 / cp.sqrt(I1[:, :, None])
        SV2 = SV2 / cp.sqrt(I2[:, :, None])
        SV1 = SV1 + SV2
        SV2 = SV1 - 2 * SV2
        nna = cp.sqrt((SV1 * SV1).sum(axis=2))
        nnb = cp.sqrt((SV2 * SV2).sum(axis=2))
        SV1 = SV1 / nna[:, :, None]
        SV2 = SV2 / nnb[:, :, None]

        I1, I2, If, nna, nnb = None, None, None, None, None

        SV1minus = cp.roll(SV1, -self.settings['zOffset'], axis=0)
        SV1 = cp.roll(SV1, self.settings['zOffset'], axis=0)
        SV2minus = cp.roll(SV2, -self.settings['zOffset'], axis=0)
        SV2 = cp.roll(SV2, self.settings['zOffset'], axis=0)

        # Normalized Cross Product
        oa = cp.cross(SV1 - SV1minus, SV2 - SV2minus, axis=2)
        den = cp.sqrt((oa * oa).sum(axis=2))
        den = cp.clip(den, a_min=1e-15, a_max=None)
        oa = oa / den[:, :, None]

        den = None

        # Calculate retsinW
        temp = (SV1minus * oa).sum(axis=2) ** 2
        temp = cp.clip(temp,a_min=1e-15,a_max=1-.0000001)
        temp2 = ((SV1minus * SV1).sum(axis=2) - temp) / (1-temp)
        retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']

        temp, temp2 = None, None

        pm = cp.sign((1 - (SV1minus * SV1).sum(axis=2)) * ((SV1minus - SV1) * (SV2minus + SV2)).sum(axis=2))
        SV2, SV1, SV1minus, SV2minus = None, None, None, None

        oa = oa * cp.expand_dims(pm, axis=2) * cp.expand_dims(retSinW, axis=2)
        retSinW = None

        oa = cp.mean(oa, axis=-1)

        return oa, dop

    def spectralBinning(self):
        """
        Compute retardance, degree of polarization, and optic axis frames from PS sensitive datasets

        Note:
            Reference:
            [1] Villiger, M., Zhang, E. Z., Nadkarni, S. K., Oh, W.-Y., Vakoc, B. J., & Bouma, B. E. (2013).
                Spectral binning for mitigation of polarization mode dispersion artifacts in catheter-based optical
                frequency domain imaging. Optics Express, 21(14), 16353.
                https://doi.org/10.1364/oe.21.016353

            This implementation uses spectral binning to improve the quality of PS images by decreasing polarization
            mode dispersion (PMD).
            This time to execute is proportional to the spectral bins being used.
        Args:
        Output:
            self.processedData['ret'] (np.array) : retardance (100 deg/um )
            self.processedData['dop'] (np.array) : degree of polarization
            self.processedData['oa'] (np.array) : apparent optic axis
        """

        SV1 = cp.asarray(self.sv1[:, :, 1:4, :])
        SV2 = cp.asarray(self.sv2[:, :, 1:4, :])

        # i^2 = q^2+u^2+v^2 (eq.3.36 Theocaris)>>
        I1 = cp.sqrt((SV1 * SV1).sum(axis=2))
        I2 = cp.sqrt((SV2 * SV2).sum(axis=2))

        # Filter
        SV1 = convolve(SV1, self.filter[:, :, None, None], mode='constant')
        SV2 = convolve(SV2, self.filter[:, :, None, None], mode='constant')
        I1 = convolve(I1, self.filter[:, :, None], mode='constant')
        I2 = convolve(I2, self.filter[:, :, None], mode='constant')

        If = cp.mean(I1 * I1 + I2 * I2, axis=-1)

        # Euclidian length of Q,U,V <<after averaging, so different from I1,I2>>
        I1 = (SV1 * SV1).sum(axis=2)
        I2 = (SV2 * SV2).sum(axis=2)
        # Computation of DOP / uniformity

        If = cp.clip(If, a_min=1e-15, a_max=None)
        dop = cp.sqrt(cp.mean(I1 + I2, axis=-1) / If)
        dopMask = (dop > self.settings['dopThresh']) * (dop <= 1)

        # Normalize
        SV1 = SV1 / cp.sqrt(I1[:, :, None])
        SV2 = SV2 / cp.sqrt(I2[:, :, None])

        # force the two Stokes vectors to be orthogonal, equivalent to LSQ solution
        SV1 = SV1 + SV2
        SV2 = SV1 - 2 * SV2
        nna = cp.sqrt((SV1 * SV1).sum(axis=2))
        nnb = cp.sqrt((SV2 * SV2).sum(axis=2))
        SV1 = SV1 / nna[:, :, None]
        SV2 = SV2 / nnb[:, :, None]

        I1, I2, If, nna, nnb = None, None, None, None, None

        SV1minus = cp.roll(SV1, -self.settings['zOffset'], axis=0)
        SV1 = cp.roll(SV1, self.settings['zOffset'], axis=0)
        SV2minus = cp.roll(SV2, -self.settings['zOffset'], axis=0)
        SV2 = cp.roll(SV2, self.settings['zOffset'], axis=0)

        # Normalized Cross Product
        oa = cp.cross(SV1 - SV1minus, SV2 - SV2minus, axis=2)
        den = cp.sqrt((oa * oa).sum(axis=2))
        den = cp.clip(den, a_min=1e-15, a_max=None)
        oa = oa / den[:, :, None]

        den = None

        # # Calculate retsinW
        temp = (SV1minus * oa).sum(axis=2) ** 2
        temp = cp.clip(temp, a_min=1e-15, a_max=1-.0000001)
        temp2 = ((SV1minus * SV1).sum(axis=2) - temp) / (1-temp)
        retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']
        #
        # Calculating retsinW
        # temp = cp.einsum('ijkl,ijkl->ijl', SV1minus, oa) ** 2
        # temp2 = (cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1) - temp) / (1 - temp)
        # retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']

        temp, temp2 = None, None

        pm = cp.sign((1 - cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1)) *
                     cp.einsum('ijkl,ijkl->ijl', (SV1minus - SV1), (SV2minus + SV2)))

        # pm = cp.sign((1 - (SV1minus * SV1).sum(axis=2)) * ((SV1minus - SV1) * (SV2minus + SV2)).sum(axis=2))
        oa = oa * cp.expand_dims(pm, axis=2)
        oaw = oa * cp.expand_dims(retSinW, axis=2)

        SV2, SV1, SV1minus, SV2minus, retSinW = None, None, None, None, None

        oa = cp.nan_to_num(oa)
        oaw = cp.nan_to_num(oaw)

        # Editted from MATLAB script to take mean of correction matrix rather than doing SVD on each aline
        mid = int(oa.shape[-1] / 2)
        ref = oa[:, :, :, mid]
        oa = oa * cp.expand_dims(dopMask[:, :, None], axis=3)
        C = cp.zeros((3, 3, oa.shape[1]))
        diagArr = cp.identity(3)
        for wind in range(oa.shape[-1]):
            for i in range(oa.shape[2]):
                for j in range(oa.shape[2]):
                    C[j, i, :] = cp.sum(oa[:, :, i, wind] * ref[:, :, j], axis=0)
            Cha = C.mean(axis=2)
            [u, s, vh] = cp.linalg.svd(Cha)
            diagArr[2, 2] = cp.linalg.det(cp.matmul(u, vh))
            Rha = cp.matmul(cp.matmul(u, diagArr), vh)
            oaw[:, :, :, wind] = cp.einsum('ijl,ml->ijm', oaw[:, :, :, wind], Rha)

        u, s, vh, wind, tempn1, tempn2, tempn3 = None, None, None, None, None, None, None
        ref, dopMask, pm, retSinW = None, None, None, None

        rpamean = cp.mean(oaw, axis=-1)
        return rpamean, dop

    def symmetric(self):
        """
        Compute retardance, degree of polarization, and optic axis frames from PS sensitive datasets

        Note:
            Reference:
            [1] Villiger, M., Zhang, E. Z., Nadkarni, S. K., Oh, W.-Y., Vakoc, B. J., & Bouma, B. E. (2013).
                Spectral binning for mitigation of polarization mode dispersion artifacts in catheter-based optical
                frequency domain imaging. Optics Express, 21(14), 16353.
                https://doi.org/10.1364/oe.21.016353

            [2] Li, Q., Karnowski, K., Noble, P. B., Cairncross, A., James, A., Villiger, M., & Sampson, D. D. (2018).
                Robust reconstruction of local optic axis orientation with fiber-based polarization-sensitive
                optical coherence tomography. Biomedical Optics Express, 9(11), 5437–5455.
                https://doi.org/10.1364/BOE.9.005437

            This implementation uses spectral binning to improve the quality of PS images by decreasing polarization
            mode dispersion (PMD).
            This time to execute is proportional to the spectral bins being used.
        Args:
        Output:
            self.processedData['ret'] (np.array) : retardance (100 deg/um )
            self.processedData['dop'] (np.array) : degree of polarization
            self.processedData['oa'] (np.array) : apparent optic axis
        """

        SV1 = cp.asarray(self.sv1[:, :, 1:4, :])
        SV2 = cp.asarray(self.sv2[:, :, 1:4, :])

        # i^2 = q^2+u^2+v^2 (eq.3.36 Theocaris)>>
        I1 = cp.sqrt((SV1 * SV1).sum(axis=2))
        I2 = cp.sqrt((SV2 * SV2).sum(axis=2))

        # Filter
        SV1 = convolve(SV1, self.filter[:, :, None, None], mode='constant')
        SV2 = convolve(SV2, self.filter[:, :, None, None], mode='constant')
        I1 = convolve(I1, self.filter[:, :, None], mode='constant')
        I2 = convolve(I2, self.filter[:, :, None], mode='constant')

        If = cp.mean(I1 * I1 + I2 * I2, axis=-1)

        # Euclidian length of Q,U,V <<after averaging, so different from I1,I2>>
        I1 = (SV1 * SV1).sum(axis=2)
        I2 = (SV2 * SV2).sum(axis=2)
        # Computation of DOP / uniformity

        If = cp.clip(If, a_min=1e-15, a_max=None)
        dop = cp.sqrt(cp.mean(I1 + I2, axis=-1) / If)
        dopMask = (dop > self.settings['dopThresh']) * (dop <= 1)

        # Normalize
        SV1 = SV1 / cp.sqrt(I1[:, :, None])
        SV2 = SV2 / cp.sqrt(I2[:, :, None])

        # force the two Stokes vectors to be orthogonal, equivalent to LSQ solution
        SV1 = SV1 + SV2
        SV2 = SV1 - 2 * SV2
        nna = cp.sqrt((SV1 * SV1).sum(axis=2))
        nnb = cp.sqrt((SV2 * SV2).sum(axis=2))
        SV1 = SV1 / nna[:, :, None]
        SV2 = SV2 / nnb[:, :, None]

        I1, I2, If, nna, nnb = None, None, None, None, None

        SV1minus = cp.roll(SV1, -self.settings['zOffset'], axis=0)
        SV1 = cp.roll(SV1, self.settings['zOffset'], axis=0)
        SV2minus = cp.roll(SV2, -self.settings['zOffset'], axis=0)
        SV2 = cp.roll(SV2, self.settings['zOffset'], axis=0)

        # Normalized Cross Product
        oa = cp.cross(SV1 - SV1minus, SV2 - SV2minus, axis=2)
        den = cp.sqrt((oa * oa).sum(axis=2))
        den = cp.clip(den, a_min=1e-15, a_max=None)
        oa = oa / den[:, :, None]

        den = None

        # # Calculate retsinW
        temp = (SV1minus * oa).sum(axis=2) ** 2
        temp = cp.clip(temp,a_min=1e-15,a_max=1-.0000001)
        temp2 = ((SV1minus * SV1).sum(axis=2) - temp) / (1-temp)
        retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']
        #
        # Calculating retsinW
        # temp = cp.einsum('ijkl,ijkl->ijl', SV1minus, oa) ** 2
        # temp2 = (cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1) - temp) / (1 - temp)
        # retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / self.settings['zOffset']

        temp, temp2 = None, None

        pm = cp.sign((1 - cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1)) *
                     cp.einsum('ijkl,ijkl->ijl', (SV1minus - SV1), (SV2minus + SV2)))

        # pm = cp.sign((1 - (SV1minus * SV1).sum(axis=2)) * ((SV1minus - SV1) * (SV2minus + SV2)).sum(axis=2))
        oa = oa * cp.expand_dims(pm, axis=2)
        oaw = oa * cp.expand_dims(retSinW, axis=2)

        SV2, SV1, SV1minus, SV2minus, retSinW = None, None, None, None, None

        oa = cp.nan_to_num(oa)
        oaw = cp.nan_to_num(oaw)

        mid = int(oa.shape[-1] / 2)
        if not (self.corrections['symmetry'] is None) and not (self.corrections['bins'] is None):
            for wind in range(oa.shape[-1]):
                if self.corrections['symmetry'].shape[-1] > 1:
                    temp = self.corrections['symmetry'][:, :, wind] @ self.corrections['bins'][:, :, wind]
                else:
                    temp = self.corrections['symmetry'][:, :, 0] @ self.corrections['bins'][:, :, wind]

                oaw[:, :, :, wind] = cp.einsum('ijl,ml->ijm', oaw[:, :, :, wind], temp)

        else:
            ref = oa[:, :, :, mid]
            oa = oa * cp.expand_dims(dopMask[:, :, None], axis=3)
            C = cp.zeros((3, 3, oa.shape[1]))
            diagArr = cp.identity(3)
            R = cp.zeros((3, 3, oa.shape[-1]))
            for wind in range(oa.shape[-1]):
                for i in range(oa.shape[2]):
                    for j in range(oa.shape[2]):
                        C[j, i, :] = cp.sum(oa[:, :, i, wind] * ref[:, :, j], axis=0)
                Cha = C.mean(axis=2)
                [u, s, vh] = cp.linalg.svd(Cha)
                if wind == mid and (self.corrections['symmetry'] is None):
                    self.corrections['symmetry'] = vh[:, :, None]
                diagArr[2, 2] = cp.linalg.det(cp.matmul(u, vh))
                R[:, :, wind] = cp.matmul(cp.matmul(u, diagArr), vh)
                #oaw[:, :, :, wind] = cp.einsum('ijl,ml->ijm', oaw[:, :, :, wind], R[:, :, wind])
            if (self.corrections['bins'] is None):
                self.corrections['bins'] = R

            for wind in range(oa.shape[-1]):
                if self.corrections['symmetry'].shape[-1] > 1:
                    temp = self.corrections['symmetry'][:, :, wind] @ self.corrections['bins'][:, :, wind]
                else:
                    temp = self.corrections['symmetry'][:, :, 0] @ self.corrections['bins'][:, :, wind]
                oaw[:, :, :, wind] = cp.einsum('ijl,ml->ijm', oaw[:, :, :, wind], temp)

        u, s, vh, wind, tempn1, tempn2, tempn3 = None, None, None, None, None, None, None
        ref, dopMask, pm, retSinW = None, None, None, None

        rpamean = cp.mean(oaw, axis=-1)
        return rpamean, dop

    def measurementMat(self):
        """
        Compute retardance, degree of polarization, and optic axis frames from PS sensitive datasets

        Note:
            Reference:
            [1] Villiger, M., Braaf, B., Lippok, N., Otsuka, K., Nadkarni, S. K., & Bouma, B. E. (2018).
                Optic axis mapping with catheter-based polarization-sensitive optical coherence tomography.
                Optica, 5(10), 1329. https://doi.org/10.1364/optica.5.001329
        Args:
        Output:
            self.processedData['ret'] (np.array) : retardance (100 deg/um )
            self.processedData['dop'] (np.array) : degree of polarization
            self.processedData['oa'] (np.array) : apparent optic axis
        """

        SV1 = cp.asarray(self.sv1)
        SV2 = cp.asarray(self.sv2)

        MM, dop = makeMeasurementMatrix(SV1, SV2, stokesFilter=None)

        if self.corrections['symmetry'] is None:
            symRotMatrix, errInit, errOpt = getSymmetryCorrection(MM, dop)
            self.corrections['symmetry'] = symRotMatrix
        MM = applyRotation(MM, self.corrections['symmetry'])

        if self.corrections['bins'] is None:
            binRotMatrix, vhOut = getBinCorrection(MM, dop)
            self.corrections['bins'] = binRotMatrix
        MM = applyRotation(MM, self.corrections['bins'])

        oa = decomposeRot(MM)

        # Average bins and put in form [Nz,NAlines,3]
        oa = oa.mean(axis=-1).transpose(1, 2, 0)
        oa = cp.arctan(oa)

        return oa, dop

    def formatOut(self, oa, dop):
        """ Format output contrasts to uint8 """

        ret = self.formatRetOut(oa)
        dop = self.formatDopOut(dop)
        theta = self.formaThetaOut(oa)
        oa = self.formatOaOut(oa)

        return ret, dop, oa, theta

    def formatRetOut(self, oa, dn=False):
        """ Format the output ret array to numpy uint8, according to ps settings

        Notes:
            dn = True allows non-normalized deltaN birefringence
        Args:
            ret (array) : Un-formatted float retardance array - Units [deg / 100um]
        Returns:
            array : Uint8 (0-255) formatted retardance array with -  Units [deg / 100um]
        """
        ret = cp.clip(cp.sqrt((oa * oa).sum(axis=-1)), a_min=1e-15, a_max=None)
        if dn:
            ret = ret
        else:
            ret = ret * (100 / self.settings['zResolution']) * (180 / math.pi) / self.settings['maxRet'] * 255
        return cp.asnumpy(ret.astype('uint8'))

    def formatDopOut(self, dop, b_clip_dop=False):
        """ Format the output dop array to numpy uint8. Have an option of re-scaling at 0.5
        #TODO: make b_clip_dop accessible
        Args:
            dop (array) : Un-formatted float apparent degree of polarization array - Scale [0,1]
        Returns:
            array : Uint8 formatted oa array - Scale [0,255]
        """

        # Increase contrast
        if b_clip_dop:
            dop = (cp.clip(dop, a_min=0.5, a_max=None) - 0.5) / 0.5
        else:
            dop = cp.clip(dop, a_min=0, a_max=1)
        return cp.asnumpy((255 * dop).astype('uint8'))

    def formatOaOut(self, oa):
        """ Format the output oa array to numpy uint8
        Args:
            oa (array) : Un-formatted float apparent optic axis array - Scale [-pi,pi]
        Returns:
            array : Uint8 formatted oa array - Scale [0,255]
        """
        ret = cp.clip(cp.sqrt((oa * oa).sum(axis=-1)), a_min=1e-15, a_max=None)
        return cp.asnumpy((255 * (oa / ret[:, :, None] * 0.5 + 0.5)).astype('uint8'))

    def formaThetaOut(self, oa, shift=None):

        """ Format the output oa array to numpy uint8
        Args:
            oa (array) : Un-formatted float apparent optic axis array - Scale [-pi,pi]
        Returns:
            theta (array) : Uint8 formatted theta array - Scale [0,255]
        """
        if shift:
            self.settings['thetaOffset'] = shift

        complexAngle = (oa[:, :, 0] + oa[:, :, 1] * 1j)
        theta = cp.squeeze(cp.angle(complexAngle)) + self.settings['thetaOffset'] / 180 * math.pi
        theta[theta > math.pi] = -math.pi + (theta[theta > math.pi] - math.pi)
        return cp.asnumpy(((theta / math.pi / 2 + 1) * 255).astype('uint8'))

    def clearPadding(self, img):
        """ Deletes padding if exists"""
        if img.shape[1] > self.settings['imgWidth']:
            img = img[:, int(self.settings['xFilter'] * 2):int(-self.settings['xFilter'] * 2)]
        return img
