import os, sys, fnmatch, struct, cdflib, configparser, logging
import xml.etree.ElementTree as ET
from .metadata import Metadata
from .fringe import Fringe
from ..utils import *

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

def dataFormat(directory: str) -> str:
    """ Check the format of the data within a directory """
    # Easiest way to be sure there is a settings folder to check
    if not os.path.exists(os.path.join(directory, 'settings')):
        os.mkdir(os.path.join(directory, 'settings'))

    if len(fnmatch.filter(os.listdir(directory), '*.dat')) < 1:
        if len(fnmatch.filter(os.listdir(directory), '*.cfd')) < 1:
            df = 'VakocV1'
        else:
            df = 'VakocV2'

        if len(fnmatch.filter(os.listdir(os.path.join(directory, 'settings')), '*reconstructsettings*')) > 0:
            df = 'VakocVRS'

        # Override for editSettingsFile presence
        esf = os.path.join(os.path.join(directory, 'settings'), 'editsettings.ini')
        if os.path.exists(esf):
            df = 'VakocVES'
    else:
        df = 'Bouma'

        if len(fnmatch.filter(os.listdir(os.path.join(directory, 'settings')), '*reconstructsettings*')) > 0:
            df = 'BoumaVRS'
        # Override for editSettingsFile presence
        esf = os.path.join(os.path.join(directory, 'settings'), 'editsettings.ini')
        if os.path.exists(esf):
            df = 'BoumaVES'

    return df

class Dataset(Metadata, Fringe):
    """ Creates a dataset ready to be processed, from metadata and fringe data"""
    def __init__(self, mode='2channel', id = None):
        self.debugMode = 1
        self.numChannels = 2

        self.basenameInPath = ''
        self.basenameOutPath = ''
        self.processedPath = ''
        self.settingsPath = ''
        self.success = 0

        self.chirp = []
        self.dispersion = []
        self.bg1 = []
        self.bg2 = []

        # Added for GPU Programming
        self.dataRange = []

        # Make paths
        Metadata.__init__(self)
        Fringe.__init__(self, mode)

        self.inputFilenames['reconstructID']=id
        
        self.makePaths()
        self.loadMetadata()
        self.loadFringe(frame=1)
        logging.info('Fringe raw data loaded from: {}'.format(self.inputFilenames['ofd']))
        logging.info('Shape of Raw Bscan: {}'.format(self.rawBScan.shape[0]))
        logging.info('Shape of data kept: {}'.format(self.ch1.shape))
        logging.info('Shape of image volume: {},{},{}'.format(self.reconstructionSettings['numSamples'],
                                                              self.scanSettings['numAlinesPerRawFrame'],
                                                              self.numFrames))

    def initialize(self):
        """
        This function initialized or reinitializes important dependant variables after updating some variable
        """

        # Report and generate editSettings
        self.assignProcessSettings()
        self.assignScanSettings()

        if self.processOptions['correctSystemOA']:
            self.loadPsCorrections()

        self.report()
        self.generateEditSettings()

    def makePaths(self):
        """ Makes the paths for later use during the write phase of the processing"""
        filename = os.path.splitext(fnmatch.filter(os.listdir(self.directory), '*.ofd')[0])[0]

        self.processedPath = os.path.join(self.directory, 'Processed')
        if not os.path.exists(self.processedPath):
            os.mkdir(self.processedPath)

        self.basename = filename
        self.basenameInPath = os.path.join(self.directory, filename)
        self.basenameOutPath = os.path.join(self.processedPath, filename)

        self.storageSettings['basenameOutPath'] = self.basenameOutPath

        self.settingsPath = os.path.join(self.directory, 'settings')
        if not os.path.exists(self.settingsPath):
            os.mkdir(self.settingsPath)

    def assignProcessSettings(self):
        """ Assigns some processing settings to other locations where they are required """
        carrier = self.reconstructionSettings['clockRateMHz'] * float(self.reconstructionSettings['demodSet'][0]) / 2
        normDemodCarrier = carrier / (self.reconstructionSettings['clockRateMHz'] / 2)
        self.reconstructionSettings['demodCarrierIndex'] = round(normDemodCarrier * 0.5 *
                                                                 self.reconstructionSettings['numSamples'])
        self.reconstructionSettings['demodReverseIndex'] = round(self.reconstructionSettings['demodCarrierIndex'] *
                                                                 self.reconstructionSettings['numZOut'] /
                                                                 self.reconstructionSettings['numSamples'])

        self.reconstructionSettings['binFract'] = self.psSettings['binFract']
        self.reconstructionSettings['xFilter'] = self.psSettings['xFilter']
        self.reconstructionSettings['fastProcessing'] = self.processOptions['fastProcessing']
        self.reconstructionSettings['spectralBinning'] = self.processOptions['spectralBinning']

        self.psSettings['fastProcessing'] = self.processOptions['fastProcessing']
        self.psSettings['spectralBinning'] = self.processOptions['spectralBinning']
        self.psSettings['correctSystemOA'] = self.processOptions['correctSystemOA']

    def assignScanSettings(self):
        """Takes scan settings from file and disperses into respective settings sections"""
        self.reconstructionSettings['imgWidth'] = self.scanSettings['imgWidth']
        self.reconstructionSettings['imgDepth'] = self.scanSettings['imgDepth']

        self.structureSettings['imgDepth'] = self.scanSettings['imgDepth']

        self.angioSettings['imgWidthAng'] = self.scanSettings['imgWidthAng']
        self.angioSettings['imgDepthAng'] = self.scanSettings['imgDepthAng']
        self.angioSettings['imgDepth'] = self.scanSettings['imgDepth']
        self.angioSettings['AlinesToProcAngioLinesA'] = self.scanSettings['AlinesToProcAngioLinesA']
        self.angioSettings['AlinesToProcAngioLinesB'] = self.scanSettings['AlinesToProcAngioLinesB']

        self.psSettings['imgWidth'] = np.int(self.scanSettings['imgWidth']/2)

    def report(self):
        """Reports settings information to the log file"""

        logging.info('====================================================')
        logging.info('Data information:')
        logging.info('Output will be stored in: {}'.format(self.basenameOutPath))
        logging.info('Zoom Factor: {}'.format(self.reconstructionSettings['zoomFactor']))
        logging.info('Number of sampling points: {}'.format(self.reconstructionSettings['numSamples']))
        logging.info('Image width: {}'.format(self.scanSettings['imgWidth']))
        logging.info('Image width angio: {}'.format(self.scanSettings['imgWidthAng']))
        logging.info('Image depth: {}'.format(self.scanSettings['imgDepth']))
        logging.info('Image depth angio: {}'.format(self.scanSettings['imgDepthAng']))
        logging.info('Image height: {}'.format(self.reconstructionSettings['numZOut']))
        logging.info('Number of frames: {}'.format(self.scanSettings['numFrames']))
        logging.info('====================================================')

        if self.debugMode:
            logging.info('Processing options:')
            for i in self.processOptions.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.processOptions[i]))
            logging.info('====================================================')
            logging.info('Reconstruction Settings:')
            for i in self.reconstructionSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.reconstructionSettings[i]))
            logging.info('====================================================')
            logging.info('Scan Pattern:')
            for i in self.scanSettings.keys():
                if i == 'AlinesToProcTomo':
                    pass
                else:
                    logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.scanSettings[i]))
            logging.info('====================================================')
            logging.info('Input filesnames according to settings.ini:')
            for i in self.inputFilenames.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.inputFilenames[i]))
            logging.info('====================================================')
            logging.info('Struct settings:')
            for i in self.structureSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.structureSettings[i]))
            logging.info('====================================================')
            logging.info('Angio settings:')
            for i in self.angioSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.angioSettings[i]))
            logging.info('====================================================')
            logging.info('Ps settings:')
            for i in self.psSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.psSettings[i]))
            logging.info('====================================================')
            logging.info('HSV settings:')
            for i in self.hsvSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.hsvSettings[i]))
            logging.info('====================================================')
            logging.info('Storage settings:')
            for i in self.storageSettings.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.storageSettings[i]))
            logging.info('====================================================')
            logging.info('Ps corrections:')
            for i in self.psCorrections.keys():
                logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=self.psCorrections[i]))
            logging.info('====================================================')

    def loadXml(self):
        """
        Loads .xml files

        Args:
            self (obj) : For main directory string
        """
        filename = fnmatch.filter(os.listdir(self.directory), '*.xml')
        self.inputFilenames['config'] = os.path.join(self.directory, filename[0])
        logging.info('Acquisition info loaded from: {}'.format(self.inputFilenames['config']))
        tree = ET.parse(self.inputFilenames['config'])
        root = tree.getroot()
        info = {}

        for child in root.iter():
            info[child.tag] = child.text

        # Keep only important variables
        self.xml = info
        self.parseXml()

    def loadReconstructSettings(self):
        """
        Loads some reconstruction information from Reconstructsetting.ini files

        Args:
            self (obj) : For main directory string
        """

        config = configparser.ConfigParser()

        if (self.inputFilenames['reconstructID'] is None):
            logging.info('Reconstruct ID # : {}'.format('None'))
            ReconstructString = 'reconstructsettings.ini'
            logging.info('Reconstruct filename queried: {}'.format(ReconstructString))
        else:
            logging.info('Reconstruct ID # : {}'.format(self.inputFilenames['reconstructID']))
            ReconstructString = 'reconstructsettings_' + str(self.inputFilenames['reconstructID']) + '.ini'
            logging.info('Reconstruct settings filename queried: {}'.format(ReconstructString))

        filename = fnmatch.filter(os.listdir(self.settingsPath), ReconstructString)
        logging.info('Reconstruct filename # : {}'.format(filename[0]))
        self.inputFilenames['config'] = os.path.join(self.settingsPath, filename[0])
        config.read(self.inputFilenames['config'])

        self.reconstructionSettings['lab'] = str(config['Version']['sLab'])
        self.reconstructionSettings['version'] = str(config['Version']['sVersion'])

        self.reconstructionSettings['numSamples'] = int(config['CplxTomSetting']['nSamples'])
        self.reconstructionSettings['numAlines'] = int(config['CplxTomSetting']['nLines'])
        self.reconstructionSettings['numFrames'] = int(config['CplxTomSetting']['nFrames'])
        self.reconstructionSettings['numZOut'] = int(config['CplxTomSetting']['nZscans'])
        self.reconstructionSettings['zoomFactor'] = int(config['CplxTomSetting']['nZoomLevel'])
        self.reconstructionSettings['zoomFactorRT'] = int(config['CplxTomSetting']['nZoomLevelRT'])  # To be changed
        self.reconstructionSettings['depthIndex'][0] = int(config['CplxTomSetting']['nIndexLow'])  # To be changed
        self.reconstructionSettings['depthIndex'][1] = int(config['CplxTomSetting']['nIndexHigh'])  # To be changed
        if bool(config['CplxTomSetting']['nFrameInterval']):
            self.reconstructionSettings['frameInterval'] = int(config['CplxTomSetting']['nFrameInterval'])
        self.reconstructionSettings['flipUpDown'] = bool(config['CplxTomSetting']['bFlipaline'])
        self.inputFilenames['chirp'] = str(config['CplxTomSetting']['sMappingfilename'])
        self.inputFilenames['dispersion'] = str(config['CplxTomSetting']['sDispersionfilename'])
        #self.inputFilenames['config'] = str(config['CplxTomSetting']['sConfigfilename'])
        self.inputFilenames['scanpattern'] = str(config['CplxTomSetting']['sScanpatternfilename'])
        self.reconstructionSettings['demodSet'] = [float(x.strip()) for x in
                                                            config['CplxTomSetting']['afDemodulation'][:-1].split(
                                                                ',')]
        self.reconstructionSettings['clockRateMHz'] = int(config['CplxTomSetting']['fClockRateMHz'])
        # self.numLines = int(config['CplxTomSetting']['nLines'])
        # self.swindowtype = str(config['CplxTomSetting']['swindowtype'])
        # self.bfrequencymux = bool(config['CplxTomSetting']['bfrequencymux'])

        freflow = float(config['StructTomSetting']['fReflow'])
        frefhigh = float(config['StructTomSetting']['fRefhigh'])
        self.structureSettings['REF_lowhigh'] = [freflow, frefhigh]
        self.structureSettings['invertGray'] = str2bool(config['StructTomSetting']['bInvertgray'])
        # self.nimgwidth = int(config['StructTomSetting']['nimgwidth'])
        # self.nimgheight = int(config['StructTomSetting']['nimgheight'])
        # self.nimgdepth = int(config['StructTomSetting']['nimgdepth'])
        # self.nbitdepth = int(config['StructTomSetting']['nbitdepth'])
        # self.sinterp = str(config['StructTomSetting']['sinterp'])
        # self.nnumthreads = int(config['StructTomSetting']['nnumthreads'])

        freflowA = float(config['AngioTomSetting']['fReflow'])
        frefhighA = float(config['AngioTomSetting']['fRefhigh'])
        self.angioSettings['REF_lowhigh'] = [freflowA, frefhighA]
        self.angioSettings['invertGray'] = str2bool(config['AngioTomSetting']['bInvertgray'])
        # self.nimgwidthA = int(config['AngioTomSetting']['nimgwidth'])
        # self.nimgheightA = int(config['AngioTomSetting']['nimgheight'])
        # self.nimgdepthA = int(config['AngioTomSetting']['nimgdepth'])
        # self.nbitdepthA = int(config['AngioTomSetting']['nbitdepth'])
        # self.sinterpA = str(config['AngioTomSetting']['sinterp'])
        # self.nnumthreadsA = int(config['AngioTomSetting']['nnumthreads'])

        self.psSettings['maxRet'] = int(config['PsTomSetting']['nMaxRet'])
        self.psSettings['binFract'] = int(config['PsTomSetting']['nBinFract'])
        self.psSettings['zOffset'] = int(config['PsTomSetting']['nZOffset'])
        self.psSettings['zResolution'] = float(config['PsTomSetting']['fZResolution'])
        self.psSettings['xFilter'] = int(config['PsTomSetting']['nXFilter'])
        self.psSettings['zFilter'] = int(config['PsTomSetting']['nZFilter'])
        self.psSettings['oopFilter'] = int(config['PsTomSetting']['nOopFilter'])
        self.psSettings['dopThresh'] = float(config['PsTomSetting']['fDopThresh'])
        self.psSettings['thetaOffset'] = int(config['PsTomSetting']['nThetaOffset'])

        self.hsvSettings['thetaRef'] = int(config['HSVSetting']['nThetaRef'])
        self.hsvSettings['hueCCW'] = str2bool(config['HSVSetting']['bHueCCW'])
        self.hsvSettings['opacity'] = float(config['HSVSetting']['fOpacity'])
        self.hsvSettings['hsvCrop'] = np.zeros(2, dtype='int32')
        self.hsvSettings['hsvCrop'][0] = int(config['HSVSetting']['nHsvCropLow'])
        self.hsvSettings['hsvCrop'][1] = int(config['HSVSetting']['nHsvCropHigh'])
        self.hsvSettings['dopWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['dopWeight'][0] = int(config['HSVSetting']['nDopWeightLow'])
        self.hsvSettings['dopWeight'][1] = int(config['HSVSetting']['nDopWeightHigh'])
        self.hsvSettings['structWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['structWeight'][0] = int(config['HSVSetting']['nStructWeightLow'])
        self.hsvSettings['structWeight'][1] = int(config['HSVSetting']['nStructWeightHigh'])
        self.hsvSettings['retWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['retWeight'][0] = int(config['HSVSetting']['nRetWeightLow'])
        self.hsvSettings['retWeight'][1] = int(config['HSVSetting']['nRetWeightHigh'])
        self.hsvSettings['maskThresholds'] = np.zeros(3, dtype='int32')
        self.hsvSettings['maskThresholds'][0] = int(config['HSVSetting']['nMaskThresholdsDOP'])
        self.hsvSettings['maskThresholds'][1] = int(config['HSVSetting']['nMaskThresholdsRet'])
        self.hsvSettings['maskThresholds'][2] = int(config['HSVSetting']['nMaskThresholdsStruct'])

        self.processOptions['OOPAveraging'] = str2bool(config['ProcOptions']['bOOPAveraging'])
        self.processOptions['fastProcessing'] = str2bool(config['ProcOptions']['bFastProcessing'])
        self.processOptions['spectralBinning'] = str2bool(config['ProcOptions']['bSpectralBinning'])
        self.processOptions['correctSystemOA'] = str2bool(config['ProcOptions']['bCorrectSystemOA'])
        self.processOptions['correctSystemDiat'] = str2bool(config['ProcOptions']['bCorrectSystemDiat'])
        self.processOptions['nFramesOACorr'] = [np.int(float(x.strip())) for x in config['ProcOptions']['nNFramesOACorr'][:].split(',')]
        if len(self.processOptions['nFramesOACorr']) < 2:
            self.processOptions['nFramesOACorr']=self.processOptions['nFramesOACorr'][0]
        self.processOptions['computeBackground'] = str2bool(config['ProcOptions']['bComputeBackground'])
        self.processOptions['nFramesBGCorr'] = [np.int(float(x.strip())) for x in config['ProcOptions']['nNFramesBGCorr'][:].split(',')]
        if len(self.processOptions['nFramesBGCorr']) < 2:
            self.processOptions['nFramesBGCorr']=self.processOptions['nFramesBGCorr'][0]
        self.processOptions['rotCartesianOutput'] = str2bool(config['ProcOptions']['bRotCartesianOutput'])
        self.processOptions['maskOutput'] = str2bool(config['ProcOptions']['bMaskOutput'])
        self.processOptions['generateProjections'] = str2bool(config['ProcOptions']['bGenerateProjections'])
        self.processOptions['projState'] = str(config['ProcOptions']['sProjState'])
        self.processOptions['projType'] = str(config['ProcOptions']['sProjType'])

        self.scanAxis = np.arange(0, 1, 1 / (self.reconstructionSettings['numSamples'] / 2))

        self.psSettings['fileInitialized'] = True
        logging.info('Processing settings loaded from: {}'.format(self.inputFilenames['config']))

    def generateEditSettings(self):
        """
        Generates an editsettings.ini file in the main data directory.
        Notes:
            If data is before reconstructsettings.ini time, it is useuful to have an edittable settings file generated.
        """
        config = configparser.RawConfigParser()
        config.optionxform = str
        editString = 'editsettings.ini'
        fullname = os.path.join(self.directory, editString)
        logging.info('Edit settings filename # : {}'.format(fullname))

        dictToWrite = {
            'Version': {},
            'CplxTomSetting': {},
            'StructTomSetting': {},
            'AngioTomSetting': {},
            'PsTomSetting': {},
            'HSVSetting': {},
            'ProcOptions': {}
        }
        dictToWrite['Version']['sLab'] = self.reconstructionSettings['lab']
        dictToWrite['Version']['sVersion'] = self.reconstructionSettings['version']
        dictToWrite['Version']['sState'] = self.reconstructionSettings['processState']

        dictToWrite['CplxTomSetting']['nSamples'] = self.reconstructionSettings['numSamples']
        dictToWrite['CplxTomSetting']['nLines'] = self.scanSettings['nAlinesToProcTomo']
        dictToWrite['CplxTomSetting']['nFrames'] = self.scanSettings['numFrames']
        dictToWrite['CplxTomSetting']['nZscans'] = self.reconstructionSettings['numZOut']
        dictToWrite['CplxTomSetting']['nZoomLevel'] = self.reconstructionSettings['zoomFactor']
        dictToWrite['CplxTomSetting']['nZoomLevelRT'] = self.reconstructionSettings['zoomFactorRT']
        dictToWrite['CplxTomSetting']['nIndexLow'] = self.reconstructionSettings['depthIndex'][0]
        dictToWrite['CplxTomSetting']['nIndexHigh'] = self.reconstructionSettings['depthIndex'][1]
        dictToWrite['CplxTomSetting']['nFrameInterval'] = self.reconstructionSettings['frameInterval']
        dictToWrite['CplxTomSetting']['bFlipaline'] = self.reconstructionSettings['flipUpDown']
        dictToWrite['CplxTomSetting']['sMappingfilename'] = self.inputFilenames['chirp']
        dictToWrite['CplxTomSetting']['sDispersionfilename'] = self.inputFilenames['dispersion']
        dictToWrite['CplxTomSetting']['sConfigfilename'] = self.inputFilenames['config']
        dictToWrite['CplxTomSetting']['sScanpatternfilename'] = self.inputFilenames['scanpattern']
        temp = ''
        for i in range(len(self.reconstructionSettings['demodSet'])):
            temp = temp + str(self.reconstructionSettings['demodSet'][i]) + ','
        dictToWrite['CplxTomSetting']['afDemodulation'] = temp
        dictToWrite['CplxTomSetting']['fClockRateMHz'] = self.reconstructionSettings['clockRateMHz']
        # self.numLines = int(config['CplxTomSetting']['numLines'])
        # self.swindowtype = str(config['CplxTomSetting']['swindowtype'])
        # self.bfrequencymux = bool(config['CplxTomSetting']['bfrequencymux'])

        dictToWrite['StructTomSetting']['fReflow'] = self.structureSettings['contrastLowHigh'][0]
        dictToWrite['StructTomSetting']['fRefhigh'] = self.structureSettings['contrastLowHigh'][1]
        dictToWrite['StructTomSetting']['bInvertgray'] = self.structureSettings['invertGray']
        # self.nimgwidth = int(config['StructTomSetting']['nimgwidth'])
        # self.nimgheight = int(config['StructTomSetting']['nimgheight'])
        # self.nimgdepth = int(config['StructTomSetting']['nimgdepth'])
        # self.nbitdepth = int(config['StructTomSetting']['nbitdepth'])
        # self.sinterp = str(config['StructTomSetting']['sinterp'])
        # self.nnumthreads = int(config['StructTomSetting']['nnumthreads'])

        dictToWrite['AngioTomSetting']['fReflow'] = self.angioSettings['contrastLowHigh'][0]
        dictToWrite['AngioTomSetting']['fRefhigh'] = self.angioSettings['contrastLowHigh'][1]
        dictToWrite['AngioTomSetting']['bInvertgray'] = self.angioSettings['invertGray']
        dictToWrite['AngioTomSetting']['nXFilter'] = self.angioSettings['xFilter']
        dictToWrite['AngioTomSetting']['nZFilter'] = self.angioSettings['zFilter']
        dictToWrite['AngioTomSetting']['bInvertgray'] = self.angioSettings['invertGray']
        # self.nimgwidthA = int(config['AngioTomSetting']['nimgwidth'])
        # self.nimgheightA = int(config['AngioTomSetting']['nimgheight'])
        # self.nimgdepthA = int(config['AngioTomSetting']['nimgdepth'])
        # self.nbitdepthA = int(config['AngioTomSetting']['nbitdepth'])
        # self.sinterpA = str(config['AngioTomSetting']['sinterp'])
        # self.nnumthreadsA = int(config['AngioTomSetting']['nnumthreads'])

        dictToWrite['PsTomSetting']['nMaxRet'] = self.psSettings['maxRet']
        dictToWrite['PsTomSetting']['nBinFract'] = self.psSettings['binFract']
        dictToWrite['PsTomSetting']['nZOffset'] = self.psSettings['zOffset']
        dictToWrite['PsTomSetting']['fZResolution'] = self.psSettings['zResolution']
        dictToWrite['PsTomSetting']['nXFilter'] = self.psSettings['xFilter']
        dictToWrite['PsTomSetting']['nZFilter'] = self.psSettings['zFilter']
        dictToWrite['PsTomSetting']['nOopFilter'] = self.psSettings['oopFilter']
        dictToWrite['PsTomSetting']['fDopThresh'] = self.psSettings['dopThresh']

        dictToWrite['PsTomSetting']['nThetaOffset'] = self.psSettings['thetaOffset']

        dictToWrite['HSVSetting']['nThetaRef'] = self.hsvSettings['thetaRef']
        dictToWrite['HSVSetting']['bHueCCW'] = self.hsvSettings['hueCCW']
        dictToWrite['HSVSetting']['fOpacity'] = self.hsvSettings['opacity']
        dictToWrite['HSVSetting']['nHsvCropLow'] = self.hsvSettings['hsvCrop'][0]
        dictToWrite['HSVSetting']['nHsvCropHigh'] = self.hsvSettings['hsvCrop'][1]
        dictToWrite['HSVSetting']['nDopWeightLow'] = self.hsvSettings['dopWeight'][0]
        dictToWrite['HSVSetting']['nDopWeightHigh'] = self.hsvSettings['dopWeight'][1]
        dictToWrite['HSVSetting']['nStructWeightLow'] = self.hsvSettings['structWeight'][0]
        dictToWrite['HSVSetting']['nStructWeightHigh'] = self.hsvSettings['structWeight'][1]
        dictToWrite['HSVSetting']['nRetWeightLow'] = self.hsvSettings['retWeight'][0]
        dictToWrite['HSVSetting']['nRetWeightHigh'] = self.hsvSettings['retWeight'][1]
        dictToWrite['HSVSetting']['nMaskThresholdsDOP'] = self.hsvSettings['maskThresholds'][0]
        dictToWrite['HSVSetting']['nMaskThresholdsRet'] = self.hsvSettings['maskThresholds'][1]
        dictToWrite['HSVSetting']['nMaskThresholdsStruct'] = self.hsvSettings['maskThresholds'][2]

        dictToWrite['ProcOptions']['bOOPAveraging'] = self.processOptions['OOPAveraging']
        dictToWrite['ProcOptions']['bFastProcessing'] = self.processOptions['fastProcessing']
        dictToWrite['ProcOptions']['bSpectralBinning'] = self.processOptions['spectralBinning']
        dictToWrite['ProcOptions']['bCorrectSystemOA'] = self.processOptions['correctSystemOA']
        dictToWrite['ProcOptions']['bCorrectSystemDiat'] = self.processOptions['correctSystemDiat']
        if isinstance(self.processOptions['nFramesOACorr'], list):
            converted_list = [str(element) for element in self.processOptions['nFramesOACorr']]
            joined_string = ",".join(converted_list)
            dictToWrite['ProcOptions']['nNFramesOACorr'] = joined_string
        else:
            dictToWrite['ProcOptions']['nNFramesOACorr'] = self.processOptions['nFramesOACorr']
        dictToWrite['ProcOptions']['bComputeBackground'] = self.processOptions['computeBackground']
        if isinstance(self.processOptions['nFramesBGCorr'], list):
            converted_list = [str(element) for element in self.processOptions['nFramesBGCorr']]
            joined_string = ",".join(converted_list)
            dictToWrite['ProcOptions']['nNFramesBGCorr'] = joined_string
        else:
            dictToWrite['ProcOptions']['nNFramesBGCorr'] = self.processOptions['nFramesBGCorr']
        dictToWrite['ProcOptions']['bRotCartesianOutput'] = self.processOptions['rotCartesianOutput']
        dictToWrite['ProcOptions']['bMaskOutput'] = self.processOptions['maskOutput']
        dictToWrite['ProcOptions']['bGenerateProjections'] = self.processOptions['generateProjections']
        dictToWrite['ProcOptions']['sProjState'] = self.processOptions['projState']
        dictToWrite['ProcOptions']['sProjType'] = self.processOptions['projType']

        for key1, data1 in dictToWrite.items():
            config[key1] = {}
            for key2, data2 in data1.items():
                config[key1]["{}".format(key2)] = str(data2)

        with open(fullname, 'w') as configfile:
            config.write(configfile)
        logging.info('Edit settings generated and put in main folder')

    def loadEditSettings(self):
        """
        Loads the generated edittable settings file.

        Notes:
            This will only be loaded if it is located in the /settings directory.
            This safety mechanism of having to move the generated editsettings.ini file is on purpose.

        Args:
            self (obj) : For main directory string
        """
        config = configparser.ConfigParser()
        editString = 'editsettings.ini'
        filename = fnmatch.filter(os.listdir(self.settingsPath), editString)
        self.inputFilenames['config'] = os.path.join(self.settingsPath, filename[0])

        config.read(self.inputFilenames['config'])


        self.reconstructionSettings['lab'] = str(config['Version']['sLab'])
        self.reconstructionSettings['version'] = str(config['Version']['sVersion'])
        self.reconstructionSettings['processState'] = str(config['Version']['sState'])

        self.reconstructionSettings['numSamples'] = int(config['CplxTomSetting']['nSamples'])
        self.reconstructionSettings['numAlines'] = int(config['CplxTomSetting']['nLines'])
        self.reconstructionSettings['numFrames'] = int(config['CplxTomSetting']['nFrames'])
        self.reconstructionSettings['numZOut'] = int(config['CplxTomSetting']['nZscans'])
        self.reconstructionSettings['zoomFactor'] = int(config['CplxTomSetting']['nZoomLevel'])
        self.reconstructionSettings['zoomFactorRT'] = int(config['CplxTomSetting']['nZoomLevelRT'])  # To be changed
        self.reconstructionSettings['depthIndex'][0] = int(config['CplxTomSetting']['nIndexLow'])  # To be changed
        self.reconstructionSettings['depthIndex'][1] = int(config['CplxTomSetting']['nIndexHigh'])  # To be changed
        if bool(config['CplxTomSetting']['nFrameInterval']):
            self.reconstructionSettings['frameInterval'] = int(config['CplxTomSetting']['nFrameInterval'])
        self.reconstructionSettings['flipUpDown'] = str2bool(config['CplxTomSetting']['bFlipaline'])
        self.inputFilenames['chirp'] = str(config['CplxTomSetting']['sMappingfilename'])
        self.inputFilenames['dispersion'] = str(config['CplxTomSetting']['sDispersionfilename'])
        #self.inputFilenames['config'] = str(config['CplxTomSetting']['sConfigfilename'])
        self.inputFilenames['scanpattern'] = str(config['CplxTomSetting']['sScanpatternfilename'])
        self.reconstructionSettings['demodSet'] = [float(x.strip()) for x in
                                                            config['CplxTomSetting']['afDemodulation'][:-1].split(
                                                                ',')]
        self.reconstructionSettings['clockRateMHz'] = int(config['CplxTomSetting']['fClockRateMHz'])
        # self.numLines = int(config['CplxTomSetting']['nLines'])
        # self.swindowtype = str(config['CplxTomSetting']['swindowtype'])
        # self.bfrequencymux = bool(config['CplxTomSetting']['bfrequencymux'])

        freflow = float(config['StructTomSetting']['fReflow'])
        frefhigh = float(config['StructTomSetting']['fRefhigh'])
        self.structureSettings['contrastLowHigh'] = [freflow, frefhigh]
        self.structureSettings['invertGray'] = str2bool(config['StructTomSetting']['bInvertgray'])
        # self.nimgwidth = int(config['StructTomSetting']['nimgwidth'])
        # self.nimgheight = int(config['StructTomSetting']['nimgheight'])
        # self.nimgdepth = int(config['StructTomSetting']['nimgdepth'])
        # self.nbitdepth = int(config['StructTomSetting']['nbitdepth'])
        # self.sinterp = str(config['StructTomSetting']['sinterp'])
        # self.nnumthreads = int(config['StructTomSetting']['nnumthreads'])

        freflowA = float(config['AngioTomSetting']['fReflow'])
        frefhighA = float(config['AngioTomSetting']['fRefhigh'])
        self.angioSettings['contrastLowHigh'] = [freflowA, frefhighA]
        self.angioSettings['invertGray'] = str2bool(config['AngioTomSetting']['bInvertgray'])
        try:
            self.angioSettings['xFilter'] = int(config['AngioTomSetting']['nXFilter'])
            self.angioSettings['zFilter'] = int(config['AngioTomSetting']['nZFilter'])
        except:
            pass
        # self.nimgwidthA = int(config['AngioTomSetting']['nimgwidth'])
        # self.nimgheightA = int(config['AngioTomSetting']['nimgheight'])
        # self.nimgdepthA = int(config['AngioTomSetting']['nimgdepth'])
        # self.nbitdepthA = int(config['AngioTomSetting']['nbitdepth'])
        # self.sinterpA = str(config['AngioTomSetting']['sinterp'])
        # self.nnumthreadsA = int(config['AngioTomSetting']['nnumthreads'])

        self.psSettings['maxRet'] = int(config['PsTomSetting']['nMaxRet'])
        self.psSettings['binFract'] = int(config['PsTomSetting']['nBinFract'])
        self.psSettings['zOffset'] = int(config['PsTomSetting']['nZOffset'])
        self.psSettings['zResolution'] = float(config['PsTomSetting']['fZResolution'])
        self.psSettings['xFilter'] = int(config['PsTomSetting']['nXFilter'])
        self.psSettings['zFilter'] = int(config['PsTomSetting']['nZFilter'])
        self.psSettings['oopFilter'] = int(config['PsTomSetting']['nOopFilter'])
        self.psSettings['dopThresh'] = float(config['PsTomSetting']['fDopThresh'])
        self.psSettings['thetaOffset'] = int(config['PsTomSetting']['nThetaOffset'])

        self.hsvSettings['thetaRef'] = int(config['HSVSetting']['nThetaRef'])
        self.hsvSettings['hueCCW'] = str2bool(config['HSVSetting']['bHueCCW'])
        self.hsvSettings['opacity'] = float(config['HSVSetting']['fOpacity'])
        self.hsvSettings['hsvCrop'] = np.zeros(2, dtype='int32')
        self.hsvSettings['hsvCrop'][0] = int(config['HSVSetting']['nHsvCropLow'])
        self.hsvSettings['hsvCrop'][1] = int(config['HSVSetting']['nHsvCropHigh'])
        self.hsvSettings['dopWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['dopWeight'][0] = int(config['HSVSetting']['nDopWeightLow'])
        self.hsvSettings['dopWeight'][1] = int(config['HSVSetting']['nDopWeightHigh'])
        self.hsvSettings['structWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['structWeight'][0] = int(config['HSVSetting']['nStructWeightLow'])
        self.hsvSettings['structWeight'][1] = int(config['HSVSetting']['nStructWeightHigh'])
        self.hsvSettings['retWeight'] = np.zeros(2, dtype='int32')
        self.hsvSettings['retWeight'][0] = int(config['HSVSetting']['nRetWeightLow'])
        self.hsvSettings['retWeight'][1] = int(config['HSVSetting']['nRetWeightHigh'])
        self.hsvSettings['maskThresholds'] = np.zeros(3, dtype='int32')
        self.hsvSettings['maskThresholds'][0] = int(config['HSVSetting']['nMaskThresholdsDOP'])
        self.hsvSettings['maskThresholds'][1] = int(config['HSVSetting']['nMaskThresholdsRet'])
        self.hsvSettings['maskThresholds'][2] = int(config['HSVSetting']['nMaskThresholdsStruct'])

        self.processOptions['OOPAveraging'] = str2bool(config['ProcOptions']['bOOPAveraging'])
        self.processOptions['fastProcessing'] = str2bool(config['ProcOptions']['bFastProcessing'])
        self.processOptions['spectralBinning'] = str2bool(config['ProcOptions']['bSpectralBinning'])
        self.processOptions['correctSystemOA'] = str2bool(config['ProcOptions']['bCorrectSystemOA'])
        self.processOptions['correctSystemDiat'] = str2bool(config['ProcOptions']['bCorrectSystemDiat'])
        self.processOptions['nFramesOACorr'] = [np.int(float(x.strip())) for x in config['ProcOptions']['nNFramesOACorr'][:].split(',')]
        if len(self.processOptions['nFramesOACorr']) < 2:
            self.processOptions['nFramesOACorr']=self.processOptions['nFramesOACorr'][0]
        self.processOptions['computeBackground'] = str2bool(config['ProcOptions']['bComputeBackground'])
        self.processOptions['nFramesBGCorr'] = [np.int(float(x.strip())) for x in config['ProcOptions']['nNFramesBGCorr'][:].split(',')]
        if len(self.processOptions['nFramesBGCorr']) < 2:
            self.processOptions['nFramesBGCorr']=self.processOptions['nFramesBGCorr'][0]
        self.processOptions['rotCartesianOutput'] = str2bool(config['ProcOptions']['bRotCartesianOutput'])
        self.processOptions['maskOutput'] = str2bool(config['ProcOptions']['bMaskOutput'])
        self.processOptions['generateProjections'] = str2bool(config['ProcOptions']['bGenerateProjections'])
        self.processOptions['projState'] = str(config['ProcOptions']['sProjState'])
        self.processOptions['projType'] = str(config['ProcOptions']['sProjType'])

        if 'bouma' in self.reconstructionSettings['lab']:
            self.setScanSettings()

        self.scanAxis = np.arange(0, 1, 1 / (self.reconstructionSettings['numSamples'] / 2))
        self.psSettings['fileInitialized'] = True
        self.hsvSettings['fileInitialized'] = True
        logging.info('Processing settings loaded from: {}'.format(self.inputFilenames['config']))

    def generatePSCorrections(self):
        """
        Generates a pscorrections.ini file in the main data directory.
        Notes:
            The placement of this file in the main directory is to avoid conflicts with one previously passed through
            the settings folder.
        """
        config = configparser.RawConfigParser()
        config.optionxform = str
        correctionString = 'pscorrections.ini'
        fullname = os.path.join(self.directory, correctionString)

        dictToWrite = {
            'Corrections' : {}
        }

        if self.psCorrections['symmetry'] is None:
            dictToWrite['Corrections']['afSymmetry'] =''
        else:
            dictToWrite['Corrections']['afSymmetry'] = ','.join(map(str, self.psCorrections['symmetry'].ravel()))

        if self.psCorrections['diattenuation'] is None:
            dictToWrite['Corrections']['afDiattenuation'] = ''
        else:
            dictToWrite['Corrections']['afDiattenuation'] = ','.join(map(str, self.psCorrections['diattenuation'].ravel()))

        if self.psCorrections['bins'] is None:
            dictToWrite['Corrections']['afBins'] =''
        else:
            dictToWrite['Corrections']['afBins'] = ','.join(map(str, self.psCorrections['bins'].ravel()))

        dictToWrite['Corrections']['nBins'] = self.psCorrections['numBins']

        for key1, data1 in dictToWrite.items():
            config[key1] = {}
            for key2, data2 in data1.items():
                config[key1]["{}".format(key2)] = str(data2)

        with open(fullname, 'w') as psFile:
            config.write(psFile)

        logging.info('PS corrections file  generated: {}'.format(fullname))

    def loadPsCorrections(self):
        """
        Loads the PS corrections file.

        Notes:
            This will only be loaded if it is located in the /settings directory.
            This safety mechanism of having to move the generated pscorrections.ini file is on purpose.

        """
        try:
            config = configparser.ConfigParser()
            correctionString = 'pscorrections.ini'
            filename = fnmatch.filter(os.listdir(self.settingsPath), correctionString)
            self.inputFilenames['psCorr'] = os.path.join(self.settingsPath, filename[0])

            config.read(self.inputFilenames['psCorr'])

            if len(config['Corrections']['afSymmetry']) > 0:
                temp = [float(x.strip()) for x in config['Corrections']['afSymmetry'][:-1].split(',')]
                self.psCorrections['symmetry'] = np.asarray(temp).reshape(3, 3)[:, :, None]
            else:
                self.psCorrections['symmetry'] = None
            if len(config['Corrections']['afDiattenuation']) > 0:
                temp = [float(x.strip()) for x in config['Corrections']['afDiattenuation'][:-1].split(',')]
                self.psCorrections['diattenuation'] = np.asarray(temp).reshape(2, 2)
            else:
                self.psCorrections['diattenuation'] = None
            if len(config['Corrections']['afBins']) > 0:
                self.psCorrections['numBins'] = np.int(config['Corrections']['nBins'])
                temp = [float(x.strip()) for x in config['Corrections']['afBins'][:-1].split(',')]
                self.psCorrections['bins'] = np.asarray(temp).reshape(3, 3, self.psCorrections['numBins'])
            else:
                self.psCorrections['bins'] = None

            self.psCorrections['numBins'] = config['Corrections']['nBins']
            logging.info('PS corrections file loaded from: {}'.format(self.inputFilenames['psCorr']))

            self.psCorrections['fileInitialized'] = 1
        except:
            logging.info('Loading PS corrections failed.')

class VakocDataset(Dataset):
    """ Load metadata for Vakoc format data"""
    def __init__(self, directory, df = 'Vakoc', id=None):
        self.directory = directory
        self.df = df
        self.lab = 'vakoc'
        super(VakocDataset, self).__init__(id=id)


    def loadMetadata(self):
        """Load all metadata into the loader"""

        self.loadingPathway = ''
        self.settingsPath = os.path.join(self.directory, 'settings')
        self.processedPath = os.path.join(self.directory, 'Processed')

        if self.df == 'VakocV1':
            self.loadScanSettings()
            self.loadXml()
            self.parseXml()
            self.loadChirpData()
            self.loadDisperionData()
            self.loadingPathway = self.loadingPathway + '.scanpattern+.xml+.laser+.dispersion'

        elif self.df == 'VakocV2':
            self.loadScanSettingsCDF()
            self.loadReconstructSettings()
            self.loadChirpData()
            self.loadDisperionData()
            self.loadingPathway = self.loadingPathway + '.cdf+.ini+.laser+.dispersion'

        elif self.df == 'VakocVES': #Version Edit Settings
            if len(fnmatch.filter(os.listdir(self.directory), '*.cfd')) < 1:
                self.loadScanSettings()
                self.loadingPathway = self.loadingPathway + '.scanpattern'
            else:
                self.loadScanSettingsCDF()
                self.loadingPathway = self.loadingPathway + '.cdf'
            self.loadEditSettings()
            self.loadChirpData()
            self.loadDisperionData()
            self.loadingPathway = self.loadingPathway + '.esini+.laser+.dispersion'

        elif self.df == 'VakocVRS': #Version Reconstruct Settings
            if len(fnmatch.filter(os.listdir(self.directory), '*.cfd')) < 1:
                self.loadScanSettings()
                self.loadingPathway = self.loadingPathway + '.scanpattern'
            else:
                self.loadScanSettingsCDF()
                self.loadingPathway = self.loadingPathway + '.cdf'
            self.loadReconstructSettings()
            self.loadChirpData()
            self.loadDisperionData()
            self.loadingPathway = self.loadingPathway + '.rsini+.laser+.dispersion'

        self.loadBackgroundData()
        self.initialize()

    def loadScanSettings(self):
        """
        Loads legacy scan pattern information from .scanpattern files

        Args:
            self (obj) : For main directory string
        """
        filename = fnmatch.filter(os.listdir(self.directory), '*.scanpattern')
        fullname = os.path.join(self.directory, filename[0])
        logging.info('Scan info loaded from: {}'.format(filename[0]))
        self.inputFilenames['scanpattern'] = fullname

        f = open(fullname, 'rb')
        scanInfo = {}
        scanInfo['AlinesPerBlock'] = struct.unpack('i', f.read(4))[0]  # [0 fread(fid,1,'int32');
        scanInfo['Angiography_Flag'] = struct.unpack('i', f.read(4))[0]
        scanInfo['nGalvoScans'] = struct.unpack('i', f.read(4))[0]
        scanInfo['xScanLength'] = struct.unpack('i', f.read(4))[0]
        scanInfo['xScanLinger'] = struct.unpack('i', f.read(4))[0]
        xscan = scanInfo['AlinesPerBlock'] * scanInfo['xScanLength']
        floatArray = []
        for i in range(int(xscan)):
            floatArray.append(struct.unpack('f', f.read(4))[0])
        scanInfo['xscan'] = floatArray
        scanInfo['yScanLength'] = struct.unpack('i', f.read(4))[0]
        scanInfo['yScanLinger'] = struct.unpack('i', f.read(4))[0]
        yscan = scanInfo['AlinesPerBlock'] * scanInfo['yScanLength']
        floatArray = []
        for i in range(int(yscan)):
            floatArray.append(struct.unpack('f', f.read(4))[0])
        scanInfo['yscan'] = floatArray
        scanInfo['nFrames'] = struct.unpack('i', f.read(4))[0]
        intArray = []
        for i in range(scanInfo['nFrames']):
            intArray.append(struct.unpack('i', f.read(4))[0])
        scanInfo['frameLocations'] = intArray
        scanInfo['framesUniform'] = struct.unpack('i', f.read(4))[0]
        scanInfo['frameLength'] = struct.unpack('H', f.read(2))[0]
        scanInfo['nAlinesToProcTomo'] = struct.unpack('H', f.read(2))[0]
        intArray = []
        for i in range(scanInfo['nAlinesToProcTomo']):
            intArray.append(struct.unpack('H', f.read(2))[0])
        scanInfo['AlinesToProcTomo'] = intArray
        scanInfo['imgWidthStr'] = struct.unpack('H', f.read(2))[0]
        scanInfo['imgDepthStr'] = struct.unpack('H', f.read(2))[0]
        intArray = []
        for i in range(scanInfo['imgWidthStr']):
            intArray.append(struct.unpack('H', f.read(2))[0])
        scanInfo['StrMapping'] = intArray
        scanInfo['nAlinesToProcAngio'] = struct.unpack('H', f.read(2))[0]
        intArray = []
        for i in range(scanInfo['nAlinesToProcAngio']):
            intArray.append(struct.unpack('H', f.read(2))[0])
        scanInfo['AlinesToProcAngio1'] = intArray
        intArray = []
        for i in range(scanInfo['nAlinesToProcAngio']):
            intArray.append(struct.unpack('H', f.read(2))[0])
        scanInfo['AlinesToProcAngio2'] = intArray
        scanInfo['imgWidthAng'] = struct.unpack('H', f.read(2))[0]
        scanInfo['imgDepthAng'] = struct.unpack('H', f.read(2))[0]
        intArray = []
        for i in range(scanInfo['imgWidthAng']):
            intArray.append(struct.unpack('H', f.read(2))[0])
        scanInfo['AngMapping'] = intArray
        f.close()

        # Store
        self.scanSettings['imgWidth'] = int(scanInfo['imgWidthStr'])
        self.scanSettings['imgDepth'] = int(scanInfo['imgDepthStr'])
        self.scanSettings['nAlinesToProcTomo'] = int(scanInfo['nAlinesToProcTomo'])
        self.scanSettings['AlinesToProcTomo'] = np.asarray(scanInfo['AlinesToProcTomo'])
        self.scanSettings['nAlinesToProcAngio'] = int(scanInfo['nAlinesToProcAngio'])
        self.scanSettings['AlinesToProcAngioLinesA'] = np.asarray(scanInfo['AlinesToProcAngio1'])
        self.scanSettings['AlinesToProcAngioLinesB'] = np.asarray(scanInfo['AlinesToProcAngio2'])
        self.scanSettings['imgWidthAng'] = int(scanInfo['imgWidthAng'])
        self.scanSettings['imgDepthAng'] = int(scanInfo['imgDepthAng'])
        self.scanSettings['AlinesPerBlock'] = int(scanInfo['AlinesPerBlock'])
        self.scanSettings['frameLength'] = int(scanInfo['frameLength'])
        self.scanSettings['numFrames'] = int(scanInfo['nFrames'])

        self.scanSettings['numAlinesPerRawFrame'] = np.int64(
            self.scanSettings['AlinesPerBlock'] * self.scanSettings['frameLength'])

    def loadScanSettingsCDF(self):
        """
        Loads scan pattern information from .cdf files

        Args:
            self (obj) : For main directory string
        """
        filename = fnmatch.filter(os.listdir(self.directory), '*.process.cdf')
        fullname = os.path.join(self.directory, filename[0])
        self.inputFilenames['scanpattern'] = fullname

        cdf_file = cdflib.CDF(fullname)
        self.scanSettings['AlinesPerBlock'] = int(cdf_file.varget('alinesperblock'))
        self.scanSettings['nAlineBlocks'] = int(cdf_file.varget('nalineblocks'))
        self.scanSettings['numFrames'] = int(cdf_file.varget('nframes'))
        self.scanSettings['frameLength'] = int(cdf_file.varget('framelength'))
        self.scanSettings['nAlinesToProcTomo'] = int(cdf_file.varget('nalinestoprocess'))
        self.scanSettings['AlinesToProcTomo'] = cdf_file.varget('alinestoprocessindex')[0].tolist()
        self.scanSettings['imgWidth'] = int(cdf_file.varget('imgwidthstr'))
        self.scanSettings['imgDepth'] = int(cdf_file.varget('imgdepthstr'))
        self.scanSettings['nAlinesToProcAngio'] = int(cdf_file.varget('nalinestoprocessangio'))
        self.scanSettings['AlinesToProcAngioLinesA'] = np.asarray(
            np.asarray(cdf_file.varget('alinestoprocessangiofirstindex'))[0].tolist())
        self.scanSettings['AlinesToProcAngioLinesB'] = np.asarray(
            np.asarray(cdf_file.varget('alinestoprocessangiosecondindex'))[0].tolist())
        self.scanSettings['imgWidthAng'] = int(cdf_file.varget('imgwidthangio'))
        self.scanSettings['imgDepthAng'] = int(cdf_file.varget('imgdepthangio'))
        self.scanSettings['numAlinesPerRawFrame'] = np.int64(
            self.scanSettings['AlinesPerBlock'] * self.scanSettings['frameLength'])
        logging.info('CDF info loaded from: {}'.format(filename))

    def loadChirpData(self, filename=None):
        """
        Loads chirp information from .laser file

        Args:
            self (obj) : For main directory string
        """
        if filename:
            self.inputFilenames['chirp'] = filename
        else:
            self.inputFilenames['chirp'] = os.path.join(self.directory,
                                                        fnmatch.filter(os.listdir(self.directory), '*.laser')[0])
        rawChirpData = np.fromfile(self.inputFilenames['chirp'], dtype='float32')
        # for some reason, MATLAB does not include initial byte
        self.chirp = rawChirpData[1::]
        logging.info('Chirp data loaded from: {}'.format(self.inputFilenames['dispersion']))

    def loadDisperionData(self, filename=None):
        """
        Loads dispersion information from .dispersion file

        Args:
            self (obj) : For main directory name
        """
        if filename:
            self.inputFilenames['dispersion'] = filename
        else:
            self.inputFilenames['dispersion'] = os.path.join(self.directory,
                                                             fnmatch.filter(os.listdir(self.directory),
                                                                '*.dispersion')[0])
        rawDispersionData = np.fromfile(self.inputFilenames['dispersion'], dtype='float32')
        # for some reason, MATLAB does not include initial byte
        self.dispersion = rawDispersionData[1::]
        logging.info('Dispersion data loaded from: {}'.format(self.inputFilenames['dispersion']))

    def loadBackgroundData(self, filename=None):
        """
        Loads background data from .ofb file

        Args:
            self (obj) : For main directory name
        """

        if filename:
            self.inputFilenames['ofb'] = filename
        else:
            self.inputFilenames['ofb'] = os.path.join(self.directory, self.basename + '.ofb')

        rawBackground = np.fromfile(self.inputFilenames['ofb'], dtype='uint16')

        logging.info('Background data loaded from: {}'.format(self.inputFilenames['ofb']))

        bg1 = rawBackground[0:self.reconstructionSettings['numSamples']]
        bg2 = rawBackground[self.reconstructionSettings['numSamples']:2*self.reconstructionSettings['numSamples']]

        self.bg1 = np.zeros((len(bg1), 2))
        self.bg2 = np.zeros((len(bg2), 2))

        self.bg1[:,0] = bg1
        self.bg1[:,1] = bg1
        self.bg2[:,0] = bg2
        self.bg2[:,1] = bg2

    def parseXml(self):
        """
        cleans .xml files depending on the version of acquisitioning software

        Args:
            self (obj) : xml dict
        """
        clockRateFail=0
        self.reconstructionSettings['lab'] = 'vakoc'
        self.reconstructionSettings['version'] = 'legacy'
        self.reconstructionSettings['numSamples'] = int(self.xml['totalSamplesPerALinePerChannel'])
        self.reconstructionSettings['zoomFactor'] = int(self.xml['proc_zoomlevel'])

        try:
            self.reconstructionSettings['numZOut'] = int(self.xml['postproc_zscans'])
        except:
            self.reconstructionSettings['numZOut'] = int(self.xml['proc_nzscans'])

        try:
            self.reconstructionSettings['clockRateMHz'] = int(self.xml['captureClockRateMHz'])
        except:
            clockRateFail = 1
            pass

        try:
            self.reconstructionSettings['clockRateMHz'] = int(self.xml['internalCaptureClockRateMSpS'])
        except:
            pass
        try:
            self.reconstructionSettings['clockRateMHz'] = int(self.xml['externalCaptureClockSpeedMHz'])
        except:
            pass


        self.reconstructionSettings['demodSet'] = [x.strip() for x in self.xml['proc_demodulation'].split(',')]
        self.structureSettings['contrastLowHigh'] = [float(x) for x in self.xml['proc_intensityloglim'].split(',')]
        self.angioSettings['contrastLowHigh'] = [float(x) for x in self.xml['proc_angiographyloglim'].split(',')]
        self.structureSettings['invertGray'] = str2bool(self.xml['proc_intensityinvertgrayscale'])
        self.angioSettings['invertGray'] = str2bool(self.xml['proc_angiographyinvertgrayscale'])
        self.reconstructionSettings['flipUpDown'] = str2bool(self.xml['proc_flipimagealines'])

class BoumaDataset(Dataset):
    """ Load metadata for Bouma format data"""
    def __init__(self, directory, df='Bouma', id=None):
        self.directory = directory
        self.df = df
        self.lab = 'bouma'
        super(BoumaDataset, self).__init__(id=id)


    def loadMetadata(self):
        """Load all metadata into the loader"""

        self.loadingPathway = ''
        self.settingsPath = os.path.join(self.directory, 'settings')
        self.processedPath = os.path.join(self.directory, 'Processed')

        if self.df == 'Bouma':
            self.loadXml()
            self.parseXml()
            self.loadChirpAndDispersion()
            self.setScanSettings()
            self.loadingPathway = '.xml+.dat'

        elif self.df == 'BoumaVES':
            self.loadEditSettings()
            self.loadChirpAndDispersion()
            self.setScanSettings()
            self.loadingPathway = '.esini+.dat'

        elif self.df == 'BoumaVRS':
            self.loadReconstructSettings()
            self.loadChirpAndDispersion()
            self.setScanSettings()
            self.loadingPathway = '.rsini+.dat'

        self.loadBackgroundData()
        self.initialize()

    def loadChirpAndDispersion(self):
        """
        Loads chirp and dispersion info from the config.dat file for Bouma lab data

        Args:
            self (obj) : For main directory name
        """
        filename = fnmatch.filter(os.listdir(self.directory), '*.dat')
        fullname = os.path.join(self.directory, filename[0])
        self.inputFilenames['chirp'] = fullname

        numPoints = int(os.path.getsize(fullname) / 4 / 4)
        mmap = np.fromfile(fullname, dtype=np.int32, count=numPoints, offset=0)
        weight = np.fromfile(fullname, dtype=np.float32, count=numPoints, offset=numPoints * 4)

        self.chirp = mmap + 1 - weight
        self.chirp = self.chirp / np.max(self.chirp)
        self.dispersion = np.fromfile(fullname, dtype=np.float32, count=numPoints * 2,
                                             offset=numPoints * 4 * 2)
        logging.info('Mapping and Disperion data loaded from: {}'.format(filename[0]))

    def parseXml(self):
        """
        cleans .xml files depending on the version of acquisitioning software

        Args:
            self (obj) : xml dict
        """
        # Hack to check version since version is not provided
        if 'CCardiOFDISDoc' in self.xml:
            self.reconstructionSettings['lab'] = 'bouma'
            self.reconstructionSettings['version'] = 'c_cardio'
            self.reconstructionSettings['numSamples'] = int(self.xml['numSamples'])
            self.reconstructionSettings['zoomFactor'] = 2
            self.reconstructionSettings['numAlines'] = int(self.xml['numAlinesImage'])
            self.reconstructionSettings['numFrames'] = int(self.xml['numAcqFrames'])
            self.reconstructionSettings['numZOut'] = np.int(
                2 ** np.ceil(np.log2(self.reconstructionSettings['numSamples'] / 2)))
            self.reconstructionSettings['demodSet'] = ['0.5', '0', '1', '0', '0', '0']
            self.reconstructionSettings['clockRateMHz'] = 100
            self.reconstructionSettings['flipUpDown'] = 1

        elif 'ui_rotationdiallabel1' in self.xml:
            self.reconstructionSettings['lab'] = 'bouma'
            self.reconstructionSettings['version'] = 'bouma_unknown'
            self.reconstructionSettings['numSamples'] = int(self.xml['totalSamplesPerALinePerChannel'])
            self.reconstructionSettings['numAlines'] = int(self.xml['totalALinesPerProcessedBScan'])
            self.reconstructionSettings['numFrames'] = int(self.xml['totalRecordedBlocks'])
            self.reconstructionSettings['zoomFactor'] = 2
            self.reconstructionSettings['numZOut'] = np.int(
                2 ** p.ceil(np.log2(self.reconstructionSettings['numSamples'])))
            self.reconstructionSettings['demodSet'] = ['0.5', '0', '1', '0', '0', '0']
            self.reconstructionSettings['clockRateMHz'] = int(self.xml['captureClockRateMHz'])
            self.reconstructionSettings['flipUpDown'] = 1

            # self.totalALinesPerTransfer = int(self.xml['totalALinesPerTransfer'])
            # self.totalAcquisitionChannels = int(self.xml['totalAcquisitionChannels'])

        else:
            self.reconstructionSettings['lab'] = 'bouma'
            self.reconstructionSettings['version'] = 'bouma_unknown'
            self.reconstructionSettings['numSamples'] = int(self.xml['totalSamplesPerALinePerChannel'])
            self.reconstructionSettings['numAlines'] = int(self.xml['totalALinesPerProcessedBScan'])
            self.reconstructionSettings['numFrames'] = int(self.xml['totalRecordedBScans'])
            self.reconstructionSettings['zoomFactor'] = 2
            self.reconstructionSettings['numZOut'] = np.int(
                2 ** np.ceil(np.log2(self.reconstructionSettings['numSamples'])))
            self.reconstructionSettings['demodSet'] = ['0.5', '0', '1', '0', '0', '0']
            self.reconstructionSettings['clockRateMHz'] = int(self.xml['captureClockRateMHz'])
            self.reconstructionSettings['flipUpDown'] = 1

        self.setScanSettings()
        self.scanAxis = np.arange(0, 1, 1 / (self.reconstructionSettings['numSamples'] / 2))
        logging.info('Bouma xml version : {}'.format(self.reconstructionSettings['version']))

    def setScanSettings(self):
        if 'c_cardio' in self.reconstructionSettings['version']:
            self.scanSettings['imgWidth'] = self.reconstructionSettings['numAlines']
            self.scanSettings['numFrames'] = self.reconstructionSettings['numFrames']
            self.scanSettings['AlinesToProcTomo'] = np.arange(0, self.scanSettings['imgWidth'])
            self.scanSettings['nAlinesToProcTomo'] = self.scanSettings['imgWidth']
            self.scanSettings['frameLength'] = self.scanSettings['imgWidth']
            self.scanSettings['AlinesPerBlock'] = 1
            self.scanSettings['imgDepth'] = 1
            self.scanSettings['imgWidthAng'] = self.scanSettings['imgWidth'] - 1
            self.scanSettings['imgDepthAng'] = 1
            self.scanSettings['nAlinesToProcAngio'] = self.scanSettings['imgWidth'] - 1
            self.scanSettings['AlinesToProcAngioLinesA'] = np.arange(0, self.scanSettings['imgWidth'] - 1)
            self.scanSettings['AlinesToProcAngioLinesB'] = np.arange(1, self.scanSettings['imgWidth'])
        else:
            self.scanSettings['imgWidth'] = self.reconstructionSettings['numAlines']
            self.scanSettings['numFrames'] = self.reconstructionSettings['numFrames']
            self.scanSettings['AlinesToProcTomo'] = np.arange(0, self.scanSettings['imgWidth'])
            self.scanSettings['nAlinesToProcTomo'] = self.scanSettings['imgWidth']
            self.scanSettings['frameLength'] = self.scanSettings['imgWidth']
            self.scanSettings['AlinesPerBlock'] = 1
            self.scanSettings['imgDepth'] = 1
            self.scanSettings['imgWidthAng'] = self.scanSettings['imgWidth'] - 1
            self.scanSettings['imgDepthAng'] = 1
            self.scanSettings['nAlinesToProcAngio'] = self.scanSettings['imgWidth'] - 1
            self.scanSettings['AlinesToProcAngioLinesA'] = np.arange(0, self.scanSettings['imgWidth'] - 1)
            self.scanSettings['AlinesToProcAngioLinesB'] = np.arange(1, self.scanSettings['imgWidth'])

        self.scanSettings['numAlinesPerRawFrame'] = np.int64(
            self.scanSettings['AlinesPerBlock'] * self.scanSettings['frameLength'])

    def loadBackgroundData(self, filename=None):
        """
        Loads background data from .ofb file

        Args:
            self (obj) : For main directory name
        """

        if filename:
            self.inputFilenames['ofb'] = filename
        else:
            self.inputFilenames['ofb'] = os.path.join(self.directory, self.basename + '.ofb')

        rawBackground = cp.fromfile(self.inputFilenames['ofb'], dtype='uint16')
        numAlinesPerChannel = rawBackground.shape[0] / self.reconstructionSettings['numSamples'] / self.numChannels

        logging.info('Background data loaded from: {}'.format(self.inputFilenames['ofb']))

        if numAlinesPerChannel < 128:
            numAlinesPerChannel = 1

        ch1 = rawBackground[0:int(self.reconstructionSettings['numSamples'] *
                                     cp.floor(numAlinesPerChannel) * 2):2]
        ch2 = rawBackground[1:int(self.reconstructionSettings['numSamples'] *
                                     cp.floor(numAlinesPerChannel) * 2):2]
        ch1 = cp.reshape(ch1, (self.reconstructionSettings['numSamples'],
                               int(cp.floor(numAlinesPerChannel))), order="F")
        ch2 = cp.reshape(ch2, (self.reconstructionSettings['numSamples'],
                               int(cp.floor(numAlinesPerChannel))), order="F")
        bg1 = cp.mean(ch1, 1).astype(cp.float32)
        bg2 = cp.mean(ch2, 1).astype(cp.float32)
        
        use_ASN_bgpatch = True
        if use_ASN_bgpatch:
            num_samples = int(self.reconstructionSettings['numSamples'])
            ch1 = rawBackground[:num_samples]
            ch2 = rawBackground[num_samples:(2*num_samples)]
            ch1 = cp.reshape(ch1, (self.reconstructionSettings['numSamples'],
                                   int(cp.floor(numAlinesPerChannel))), order="F")
            ch2 = cp.reshape(ch2, (self.reconstructionSettings['numSamples'],
                                   int(cp.floor(numAlinesPerChannel))), order="F")
            bg1 = cp.mean(ch1, 1).astype(cp.float32)
            bg2 = cp.mean(ch2, 1).astype(cp.float32)
                
        self.bg1 = cp.zeros((len(bg1), 2))
        self.bg2 = cp.zeros((len(bg2), 2))

        self.bg1[:,0] = bg1
        self.bg1[:,1] = bg1
        self.bg2[:,0] = bg2
        self.bg2[:,1] = bg2

class Load:
    """ Check directory contents and create corresponding dataset"""
    def __new__(cls, directory, id=None, *args, **kwargs) -> Dataset:
        df = dataFormat(directory)
        logging.info('Data format detected: {}'.format(df))
        if 'Vakoc' in df:
            return VakocDataset(directory, df, id=id)
        elif 'Bouma' in df:
            return BoumaDataset(directory, df, id=id)