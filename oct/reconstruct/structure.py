from oct.utils import *
import logging
from ..load.metadata import Metadata

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

class Structure:
    """ Structure contrast OCT reconstruction """
    def __init__(self, mode='log'):

        acceptedModes = 'log+linear'
        if mode in acceptedModes:
            self.mode = mode
        else:
            self.mode = 'log'

        self.filter = None
        self.initialized = False
        self.chCount = 0

        self.tomch1 = None
        self.tomch2 = None

        self.processedData = {
            'struct': None
        }

        meta = Metadata()
        self.settings = meta.structureSettings

    def initialize(self, data=None, settings=None, filterSize=(5, 5)):
        """
        Initialize the structure reconstruction with desired setting
        Args:
            data (object) : Support and raw data holder
            settings (dict) : Manual input settings holder
            filterSize (tuple) : filter X & Z sizes
        """
        if data and settings:
            self.setSettings(data=data, settings=settings)
        elif data:
            self.setSettings(data=data)
        elif settings:
            self.setSettings(settings=settings)

        if not (filter is None):
            d1 = cp.hanning(filterSize[0])
            d2 = cp.hanning(filterSize[1])
            self.filter = cp.sqrt(cp.outer(d1, d2))
            self.filter = self.filter / cp.sum(self.filter)

        self.initialized = True
        logging.info('====================================================')
        logging.info('Structure settings initialized:')
        for i in self.settings.keys():
            logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.settings[i]))

    def setSettings(self, data=None, settings=None):
        """Extract the required settings variables from the dataset metadata"""

        if data:
            for key, val in data.structureSettings.items():
                self.settings[key] = data.structureSettings[key]

        elif settings:
            for key, val in settings.items():
                self.settings[key] = settings[key]

    def requires(self):
        """Prints out the required/optional variables to perform reconstruction"""
        print('Required:')
        print("tomch1=tomch1")
        print("\n")
        print('Optional:')
        print("tomch2=tomch2")
        print("\n")
        print('OR:')
        print("data=data")
        print("\n")
        print('Settings:')
        print("data.structureSettings['contrastLowHigh'] ( [min, max])")
        print("data.structureSettings['invertGray'] ( [0 or 1])")
        print("data.structureSettings['imgWidth'] (for big-seg scans) ")
        print("\n")
        print('For reference, all possible settings and defaults are:')
        for key, value in self.settings.items():
            print("self.settings['", key, "'] : ", value)

    def manageChannels(self):
        """ Handle case where only channel 2's are passed """
        if not (self.tomch2 is None) and (self.tomch1 is None):
            self.tomch1 = self.tomch2
            self.tomch1 = None

    def reconstruct(self, tomch1=None, tomch2=None, data=None, settings=None):
        """
        Reconstruct a structure contrast frame

        Notes:
        Args:
            tomch1 (array) : Reconstructed tomogram from channel 1
            tomch1 (array) : Reconstructed tomogram from channel 2
            data (object) : An object containing all the preloaded data from a directory for post processing
            settings (dict): The required/edittable settings to process an angio contrast frame
        Returns:
            processeData (dict) : A dictionary containing an intensity contrast frame labelled 'struct'
        """

        if not self.initialized:
            if data:
                if data:
                    self.chCount = 2
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

        if self.chCount == 2:
            struct = self.intensity2Channel()
        else:
            struct = self.intensity1Channel()

        if 'log' in self.mode:
            struct = 10 * cp.log(struct)
            self.processedData['struct'] = self.formatOut(struct)
        elif 'linear' in self.mode:
            self.processedData['struct'] = cp.asnumpy(struct)
        else:
            self.processedData['struct'] = cp.asnumpy(struct)

        return self.processedData

    def intensity2Channel(self):
        """
        Compute Intensity / Structure frame from tomogram

        Args:
            tomogram (obj):  data storage object
        Output:
            data.processedData['struct']
        """
        pch1 = cp.abs(self.tomch1) ** 2
        pch2 = cp.abs(self.tomch2) ** 2

        if self.settings['imgDepth'] > 1:
            pch1 = pch1.reshape([pch1.shape[0],
                                 int(pch1.shape[1]/self.settings['imgDepth']),
                                 self.settings['imgDepth']])
            pch2 = pch2.reshape([pch2.shape[0],
                                 int(pch2.shape[1]/self.settings['imgDepth']),
                                 self.settings['imgDepth']])

            struct = cp.sum(pch1 + pch2, axis=2) / self.settings['imgDepth']

        else:
            struct = pch1 + pch2

        pch1, pch2 = None, None

        return struct


    def intensity1Channel(self):
        """
        Compute Intensity / Structure frame from tomogram

        Args:
            data (obj):  data storage object
        Output:
            data.processedData['struct']
        """
        pch1 = cp.abs(self.tomch1) ** 2

        if self.settings['imgDepth'] > 1:
            pch1 = pch1.reshape([pch1.shape[0],
                                 int(pch1.shape[1]/self.settings['imgDepth']),
                                 self.settings['imgDepth']])
            struct = cp.sum(pch1, axis=2) / self.settings['imgDepth']

        else:
            struct = pch1

        pch1 = None

        return struct

    def formatOut(self, struct):
        """ Format the output structure array to Uint8, according to contrast settings
        
        Notes:
        
        Args:
            struct (array) : Un-formatted float intensity array
        Returns:
            array : Uint8 (0-255) formatted structure array within contrast range
        """

        struct = (struct - self.settings['contrastLowHigh'][0]) / (
                self.settings['contrastLowHigh'][1] - self.settings['contrastLowHigh'][0])
        struct = cp.clip(struct, a_min=0, a_max=1)
        if self.settings['invertGray']:
            struct = struct*-1+1

        struct = cp.asnumpy((struct * 255).astype('uint8'))

        return struct




