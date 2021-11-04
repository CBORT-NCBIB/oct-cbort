import numpy as np

class Metadata:
    """
    Store all the metadata loaded during initialization

    Notes:
        THis is also where defaults are stored.
    """
    def __init__(self):

        self.inputFilenames = {
            'config': '',
            'chirp': '',
            'dispersion': '',
            'ofb': '',
            'ofd': '',
            'scanpattern': '',
            'reconstructID': None,
        }
        
        self.reconstructionMode = {
            'tom': 'heterodyne',
            'struct': 'log',
            'angio': 'cdv',
            'ps': 'sym'
        }

        self.reconstructionSettings = {
            'lab': 'vakoc',
            'version': '',
            'processState': 'struct',

            'numSamples': 0,
            'numALines': 0,
            'numFrames': 0,
            'numZOut': 0,
            'imgWidth': 0,
            'imgDepth': 1,
            'zoomFactor': 8,
            'zoomFactorRT': 2,
            'depthIndex': [0, 0],
            'frameInterval': 1,
            # 'index': [0, 0], #this is never used and maybe redundant to 'depthIndex'
            'demodSet': [0.5, 0.0, 1.0, 0,0],
            'clockRateMHz': 100,
            'flipUpDown': 0,
            'bgRemoval': 'ofb',

            'minGPURam': 4000000000,
            'factorGPURam': 1,
            'holdOnGPURam': 1,
            'chunkPadding': 0,


            'binFract': 1,
            'xFilter': 15,
            'fastProcessing': 0,
            'spectralBinning': 0,

        }

        self.scanSettings = {
            'nAlinesToProcTomo': 0,
            'AlinesToProcTomo': [],
            'angioFlag': 0,
            'imgWidthAng': 0,
            'imgDepthAng': 0,
            'nAlinesToProcAngio': [],
            'AlinesToProcAngioLinesA': [],
            'AlinesToProcAngioLinesB': [],
            'AlinesPerBlock': 0,
            'frameLength': 0,
            'numAlinesPerRawFrame': 0,
            'frameSizeBytes': 0,
            'numFrames': 0,
        }

        self.processOptions = {
            'process': 1,
            'writeProcessed': 0,
            'OOPAveraging': 0,
            'fastProcessing': 0,
            'spectralBinning': 0,
            'correctSystemOA': 0,
            'correctSystemDiat': 0,
            'nFramesOACorr': 10,
            'computeBackground': 0,
            'nFramesBGCorr': 10,
            'rotCartesianOutput': 0,
            'maskOutput': 0,
            'generateProjections': 0,
            'projState': '',
            'projType': '',
        }

        self.structureSettings = {
            'contrastLowHigh': [-40, 130],
            'invertGray': 0,
            'imgDepth': 1,
        }

        self.angioSettings = {
            'contrastLowHigh': [-40, 130],
            'invertGray': 1,
            'imgWidthAng': 0,
            'imgDepthAng': 1,
            'xFilter': 11,
            'zFilter': 1,
            'nAlinesToProcAngio': [],
            'AlinesToProcAngioLinesA': [],
            'AlinesToProcAngioLinesB': [],
        }

        self.psSettings = {
            'imgWidth': None,
            'maxRet': 100,
            'binFract': 3,
            'zOffset': 5,
            'zResolution': 5,
            'xFilter': 11,
            'zFilter': 1,
            'oopFilter': 1,
            'dopThresh': 0.7,
            'thetaOffset': 0,
            'fastProcessing': 0,
            'spectralBinning': 0,
            'correctSystemOA': 0,
            'fileInitialized': False,
        }

        self.psCorrections = {
            'symmetry': None,
            'diattenuation': None,
            'bins': None,
            'numBins': None,
            'fileInitialized': 0
        }

        self.hsvSettings = {
            'thetaRef': 0,
            'hueCCW': 0,
            'hsvCrop': [0, 0],
            'dopWeight': [20, 130],
            'structWeight': [30, 100],
            'retWeight': [10, 100],
            'maskThresholds': [30, 30, 30],
            'opacity': 0.01,
            'fileInitialized': False,
        }

        self.storageSettings = {
            'nMetaBytes': int(1024 * 1024),
            'nBytesPerSample': 4,
            'currentFrame': 1,
            'startFrame': 1,
            'endFrame': 1,
            'numFramesToProc': 1,
            'basenamePath': '',
            'storageType': 'uint8',
            'storageTypeProj': 'uint16',
            'storageTypeStokes': 'float32',
            'storageTypeComplex': 'float32',
            'storageFileType': 'mgh',
            'mergeH5': False
        }

        # Processed data holder
        self.processedData = {
            'tomch1': 0,
            'tomch2': 0,
            'k1': 0,
            'k2': 0,
            'sv1': 0,
            'sv2': 0,
            'struct': 0,
            'angio': 0,
            'weight': 0,
            'dop': 0,
            'ret': 0,
            'oa': 0,
            'shadow': 0,
            'theta': 0,
            'hsv': 0,
            'mask': 1,
            'cweight': 1
        }


    def reportMeta(self):
        print(self.__dict__)
