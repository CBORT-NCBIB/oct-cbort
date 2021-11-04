import fnmatch
import logging
import time
import os
from oct import *

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()


class Post:
    """
    A post-processing object for managing pipeline of OCT processing after acquisition

    Notes:
        The post processing object will not run without a populated "data" object.
        This data object holds all the fringe and metadata required reconstruct the different contrasts.

        self. processFrameRange is the key method in this object. This method controls all the processing
        that is to occur based on the options set in the "data" object.

        states (ie. processStates, writeStates, holdStates, projectionStates) describe which
        type of frames to process, write, hold and project for a given frame range

        options (ie. processOptions) signify variations of processing options designated from the "data" object

    """

    def __init__(self):
        logging.info('====================================================')
        logging.info('Using GPU To process: {}'.format(gpuAvailable))
        logging.info('Error: {}'.format(e))
        logging.info('====================================================')
        self.firstRun = 1
        self.frameCount = 1
        self.initialized = False
        self.systemPSCharacterized = False
        self.systemBGCharacterized = False

        meta = Metadata()
        self.processedData = meta.processedData
        self.processOptions = meta.processOptions
        self.reconstructionSettings = meta.reconstructionSettings

        self.processState = self.reconstructionSettings['processState']

        self.processStates = {}
        for key, val in self.processedData.items():
            self.processStates[key] = 0

        self.writeStates = {}
        for key, val in self.processedData.items():
            self.writeStates[key] = 0

        self.projectionStates = {}
        for key, val in self.processedData.items():
            self.projectionStates[key] = 0

        self.writeFileTypes = {
            'mgh': 1,
            'tif': 0,
            'h5': 0,
        }

        self.holdState = ''
        self.holdProcessed = False
        self.heldData = {}

        self.maskableTypes = 'angio+ret+oa+theta+hsv'

    def setState(self):
        """
        Set the reconstruction state for automated reconstruction of frame range based on the procState string
        """
        logging.info('Processing state: {}'.format(self.procState))
        logging.info('Writing state: {}'.format(self.writeState))


        self.processStates['tomch1'] = 1
        self.processStates['tomch2'] = 1

        if 'proj' in self.procState:
            self.procState = self.procState + 'proj'
            self.processOptions['generateProjections'] = 1

        if 'tomo' in self.procState:
            if self.processOptions['writeProcessed']:
                self.writeStates['tomch1'] = 1
                self.writeStates['tomch2'] = 1

        if 'stokes' in self.procState:
            self.procState = self.procState + '+ps'
            self.processStates['sv1'] = 1
            self.processStates['sv2'] = 1
            self.procState = self.procState + '+sv1+sv2'
            if self.processOptions['writeProcessed']:
                self.writeStates['sv1'] = 1
                self.writeStates['sv2'] = 1

        if 'kspace' in self.procState:
            self.processStates['k1'] = 1
            self.processStates['k2'] = 1
            self.procState = self.procState + '+k1+k2'
            if self.processOptions['writeProcessed']:
                self.writeStates['k1'] = 1
                self.writeStates['k2'] = 1

        if 'struct' in self.procState:
            self.processStates['struct'] = 1
            if self.processOptions['writeProcessed']:
                self.writeStates['struct'] = 1
        else:
            self.processStates['struct'] = 0
            self.writeStates['struct'] = 0

        if 'angio' in self.procState:
            self.processStates['angio'] = 1
            self.processStates['weight'] = 1
            self.procState = self.procState + '+weight'
            if self.processOptions['writeProcessed']:
                self.writeStates['angio'] = 1
                self.writeStates['weight'] = 1
        else:
            self.processStates['angio'] = 0
            self.writeStates['angio'] = 0
            self.processStates['weight'] = 0
            self.writeStates['weight'] = 0

        if 'shadow' in self.procState:
            self.processStates['shadow'] = 1
            if self.processOptions['writeProcessed']:
                self.writeStates['shadow'] = 1
        else:
            self.processStates['shadow'] = 0
            self.writeStates['shadow'] = 0

        if 'oa' in self.procState:
            self.procState = self.procState+'+ps'
            self.processStates['oa'] = 1
            self.procState = self.procState + '+oa'
            if self.processOptions['writeProcessed']:
                self.writeStates['oa'] = 1

        if 'ps' in self.procState:
            self.processStates['ret'] = 1
            self.processStates['dop'] = 1
            self.processStates['theta'] = 1
            if self.processOptions['generateProjections'] or self.processOptions['maskOutput']:
                self.processStates['mask'] = 1
            self.procState = self.procState + '+ret+dop+theta'
            if self.processOptions['writeProcessed']:
                self.writeStates['ret'] = 1
                self.writeStates['dop'] = 1
                self.writeStates['theta'] = 1
        else:
            self.processStates['ret'] = 0
            self.processStates['dop'] = 0
            self.processStates['theta'] = 0
            self.writeStates['ret'] = 0
            self.writeStates['dop'] = 0
            self.writeStates['theta'] = 0

        if 'hsv' in self.procState:
            self.processStates['hsv'] = 1
            self.processStates['mask'] = 1
            self.processStates['cweight'] = 1
            self.procState = self.procState + '+hsv'
            if self.processOptions['writeProcessed']:
                self.writeStates['hsv'] = 1
        else:
            self.processStates['hsv'] = 0
            self.writeStates['hsv'] = 0

        logging.info('Processing State set: {}'.format(self.processStates))

    def processFrameRange(self, data, startFrame=1, endFrame=1, procState='angio+struct+ps', writeState=False,
                          procAll=True, hold=False, holdState='struct', rotCartesian=False, recMode=None):
        """
        Processes a desired frame range for a "data" object pointing to a specific directory.

        Note:
            Processing is done frame by frame to allow for processing on low-memory machines.

            This is the main reconstruction handler, it will process, hold and write outputs based on cmd/settings input
        Args:
            data (object) : Support and raw data holder
            startFrame (int) : Frame at which to begin reconstruction
            endFrame (int) : Frame at which to end reconstruction
            procState (str) : Assigns which types of reconstruction are to be done (ie 'tomo+struct+angio')
            writeState (bool) : Option for reconstruction withing writing (debug uses)
            procAll (bool) : Option for whole file reconstruction
            hold (bool) : Whether or not to hold the processed data in memory (debugging and new functions)
            holdState (str) : Which frames should be held in memory, give specifics (ie. 'ret+oa1')
            rotCartesian (bool) : Should the saved output be in the circular format or not. Useful for 3D Visuals
        Output:
            self.heldData : All data held for analysis in memory, initiated using "hold" and "holdState" inputs.
            self.processedData :
            Saved files in the Processed directory within main data directory
        """
        try:

            self.getFileInformation(data)
            if procAll:
                data.startFrame = 1
                data.endFrame = data.numFramesInFile
                data.numFramesToProc = np.int((data.endFrame - data.startFrame + 1) /
                                           data.reconstructionSettings['frameInterval'])
                data.frames = np.arange(data.startFrame, data.endFrame + 1,
                                        data.reconstructionSettings['frameInterval'])
            else:
                data.startFrame = startFrame
                data.endFrame = endFrame
                data.numFramesToProc = np.int((data.endFrame - data.startFrame + 1) /
                                           data.reconstructionSettings['frameInterval'])
                data.frames = np.arange(data.startFrame, data.endFrame + 1,
                                        data.reconstructionSettings['frameInterval'])
            self.processOptions = data.processOptions
            self.firstRun = 1
            self.frameCount = 1
            self.initialize(data, procState=procState, writeState=writeState)

            if recMode:
                for key, val in recMode.items():
                    data.reconstructionMode[key] = recMode[key]

            if rotCartesian:
                self.processOptions['rotCartesianOutput'] = True

            if hold:
                self.holdProcessed = True
                self.holdState = holdState
                for key, val in self.processStates.items():
                    if val and key in holdState:
                        if key == 'hsv':
                            self.heldData[key] = cp.zeros(
                                (data.hsvSettings['hsvCrop'][1] - data.hsvSettings['hsvCrop'][0],
                                 data.reconstructionSettings['imgWidth'],
                                 data.numFramesToProc, 3))
                        elif key == 'oa':
                            self.heldData[key] = cp.zeros(
                                (data.reconstructionSettings['numZOut'],
                                 data.reconstructionSettings['imgWidth'],
                                 data.numFramesToProc, 3))
                        else:
                            self.heldData[key] = cp.zeros((data.reconstructionSettings['numZOut'],
                                                           data.reconstructionSettings['imgWidth'],
                                                           data.numFramesToProc))
            else:
                self.holdProcessed = False

            logging.info('Requested frames: {}-{}'.format(data.startFrame, data.endFrame))
            logging.info('Number of frames to process in total: {}'.format(data.numFramesToProc))
            logging.info('Reconstruct mode settings: {}'.format(data.reconstructionMode))

            if data.startFrame and data.endFrame <= data.numFramesInFile:
                if self.processOptions['writeProcessed']:
                    writer = Writer()

                if self.processOptions['generateProjections'] and self.processOptions['writeProcessed']:
                    projections = Project()
                    projections.populateProjections(self)
                    self.setProjectionStates()
                    projections.populateTypes(self)
                    logging.info('Generating Projections for: {}'.format(self.projectionStates))

                # START CRAWLING THROUGH FRAMES
                for frame in data.frames:
                    t = time.time()
                    data.loadFringe(frame=frame)

                    logging.info('Processing Frame {}'.format(data.currentFrame))

                    if self.processOptions['process']:
                        self.reconstructFrame(data)
                        logging.info('Processed Frame')

                    if self.processOptions['generateProjections'] and self.processOptions['writeProcessed']:
                        self.projectFrame(projections, data)
                        logging.info('Projected Frame')

                    if self.processOptions['maskOutput'] and \
                            self.processStates['dop'] and self.processOptions['writeProcessed']:
                        self.maskFrame(data, mask=data.processedData['mask'])
                        logging.info('Masked Frame')

                    if self.processOptions['rotCartesianOutput']:
                        self.convertFrameCircularCartesian(data)
                        logging.info('Converted Frame to cartesian coordinates for rotational data')

                    if self.processOptions['writeProcessed']:
                        if self.firstRun:
                            self.writeMetaData(data, writer)
                            logging.info('Metadata initialized: {}'.format(writer.metadata))
                        self.writeFrame(writer, data)
                        logging.info('Wrote Frame to disk')

                    if self.holdProcessed:
                        self.holdFrameOnMemory()
                        logging.info('Stored Frame')

                    self.firstRun = 0
                    self.frameCount = self.frameCount + 1
                    elapsed = time.time() - t
                    logging.info(
                        'Last frame took {} minutes and {} seconds'.format(int(elapsed / 60), round(elapsed % 60, 1)))
                    logging.info('STATUS')

                if self.processOptions['generateProjections'] and self.processOptions['writeProcessed']:
                    self.writeProjection(writer, projections, data)

                data.success = 1
                if self.holdProcessed:
                    return self.heldData
            else:
                logging.error('Requested frame too large for file: Last Frame - {} , Requested frame - {}'.format(
                    data.numFramesInFile, data.endFrame))

        except Exception as e:
            logging.error('Error in processFrameRange() method', exc_info=True)
            data.success = 0
            raise (e)

    def writeMetaData(self, data, writer):
        """
        Dispatch meta data writer prior to reconstruction images
        Args:
            data (object) : Support and raw data object
            writer (object) : writer object
        """
        if 'mgh' in data.storageSettings['storageFileType']:
            if self.processOptions['writeProcessed']:
                for key, val in self.writeStates.items():
                    if val:
                        if key == 'hsv':
                            pass
                        else:
                            writer.writeMetadata(data, key)
                        logging.info('Writing {} Metadata'.format(key))
        if 'tif' in data.storageSettings['storageFileType']:
            pass
        if 'h5' in data.storageSettings['storageFileType']:
            pass

    def getFileInformation(self, data):
        """
        Get filename and file length info
        Args:
            data (object) : Support and raw data object
        """
        filename = fnmatch.filter(os.listdir(data.directory), '*.ofd')
        fullname = os.path.join(data.directory, filename[0])
        logging.info('# of samples per aline {}'.format(data.reconstructionSettings['numSamples']))
        logging.info('# Alines per raw frame {}'.format(data.scanSettings['numAlinesPerRawFrame']))

        data.scanSettings['frameSizeBytes'] = cp.int64(
            data.reconstructionSettings['numSamples'] * data.scanSettings['numAlinesPerRawFrame'] * 4)
        data.numFramesInFile = math.trunc(os.stat(fullname).st_size / data.scanSettings['frameSizeBytes'])
        logging.info('Total file size {}'.format(math.trunc(os.stat(fullname).st_size)))
        logging.info('Frame Size: {}'.format(data.scanSettings['frameSizeBytes']))
        logging.info('Number of Frames in file: {}'.format(data.numFramesInFile))

    def reconstructFrame(self, data):
        """
        Manage which computations to perform. This method is seperate for scalability in the future.
        Args:
            data (obj) : Support and raw data object
        """
        # Always compute tomogram
        out = self.tom.reconstruct(data=data)
        for key, val in out.items():
             data.processedData[key] = out[key]
        out = None

        if self.processStates['struct']:
            out = self.str.reconstruct(data=data)
            for key, val in out.items():
                data.processedData[key] = out[key].copy()
            out = None
            logging.info('Structure processed')

        if self.processStates['angio']:
            out = self.ang.reconstruct(data=data)
            for key, val in out.items():
                data.processedData[key] = out[key].copy()
            out = None
            logging.info('Angio processed')

        if self.processStates['shadow'] and self.processStates['angio']:
            # self.computeShadowRemoval(data)
            data.processedData = {'shadow': shadow}
            logging.info('Shadow processed')

        if 'ps' in self.procState or self.processStates['hsv']:

            if self.processOptions['correctSystemOA']:
                if self.firstRun == 1:
                    if data.psCorrections['fileInitialized']:
                        self.systemPSCharacterized = True
                    else:
                        self.characterizeSystemPS(data)
                        self.systemPSCharacterized = True

            if data.processOptions['OOPAveraging']:
                data.processedData['sv1'], data.processedData['sv2'] = self.oopAverageStokes(data)

            out = self.ps.reconstruct(data=data)
            for key, val in out.items():
                data.processedData[key] = out[key].copy()
            out = None

            logging.info('PS processed')

        self.unifyImageWidths(data)

        if self.processStates['mask'] or \
                (self.processOptions['generateProjections'] and self.processOptions['writeProcessed']) or \
                self.processOptions['maskOutput']:
            self.computeMaskFrame(data)
            logging.info('Mask processed')

        if self.processStates['cweight']:
            self.computeWeightFrame(data)
            logging.info('Color weight processed')
        if self.processStates['theta'] and self.processStates['hsv']:
            data.processedData['hsv'] = self.color.apply(data.processedData['theta']/255,
                                                         weight=data.processedData['cweight'],
                                                         mask=data.processedData['mask'],
                                                         cmap='hsv')
            logging.info('HSV processed')

    def unifyImageWidths(self, data):
        """
        Downsample frame widths that are larger than PS frames.

        Notes:
            Aline modulated PS info is innately downsampled by 2 due to modulated alines.
            However, bi-seg scan patterns allow us to keep the original resolution,
            so this is not always needed and will only be applied if there is a discrepancy.
        """
        # Contrasts to downsample
        dskeys = 'struct+angio+weight'
        if 'ps' in self.procState or self.processStates['hsv']:
            for key, val in data.processedData.items():
                if self.processStates[key] and key in dskeys:
                    if data.processedData[key].shape[1] > data.processedData['dop'].shape[1]:
                        interval = int(data.processedData[key].shape[1] / data.processedData['dop'].shape[1])
                        data.processedData[key] = data.processedData[key][:, 0::interval]

    def setProjectionStates(self):
        """
        Manages which projections will be performed.  Sets self.projectionStates
        """
        for key, val in self.projectionStates.items():
            if key in self.processOptions['projState']:
                self.projectionStates[key] = 1

    def projectFrame(self, projections, data):
        """
        Manages the creation of inline projections
        Args:
            projections (obj) : projection storage and appending object
            data (obj) : data holder object
        """
        for key, val in self.projectionStates.items():
            if self.projectionStates[key] and self.processStates[key]:
                projections.appendBScanProjection(data.processedData[key],
                                                  key,
                                                  data,
                                                  mask=data.processedData['mask'])
                logging.info('Projected Frame')

    def writeFrame(self, writer, data):
        """
        Manages the execution of image appending to specific files
        Args:
            data (object) : Support and raw data object
            writer (object) : writer object
        """
        for key, val in self.writeStates.items():
            if val:
                if key == 'hsv':
                    image = data.processedData[key]
                else:
                    image = data.processedData[key]
                writer.appendImage(image, data, key)

    def writeProjection(self, writer, projections, data):
        """
        Manages writing of projections. Seperate to normal writer class for plugin slacability
        Args:
            data (object) : Support and raw data object
            projections (object) : Projections object
            writer (object) : writer object
        """
        for projState, val in self.projectionStates.items():
            if val and self.processStates[projState]:
                for key, val in projections.projections[projState].items():
                    image = normalizeData(projections.projections[projState][key]) * 255 ** 2
                    outname = data.basenameOutPath + '.' + projState + '_proj_' + key
                    writer.saveImage(image, outname)

    def holdFrameOnMemory(self):
        """
        Manages what frames to hold if data is to be used in a notebook.
        """
        for key, val in self.processStates.items():
            if val and key in self.holdState:
                if key == 'hsv' or 'oa':
                    self.heldData[key][:, :, self.frameCount - 1, :] = data.processedData[key]
                else:
                    self.heldData[key][:, :, self.frameCount - 1] = data.processedData[key]

    def maskFrame(self, data, mask=None):
        """
        Creates a mask frame for PS data frames
        Args:
            data (object) : Support and raw data object
        """

        if mask is None:
            data.processedData['mask'] = np.ones((data.reconstructionSettings['numZOut'],
                                                   data.reconstructionSettings['imgWidth']))

        for key, val in self.processStates.items():
            if val:
                if key in self.maskableTypes:
                    if key in 'hsv + oa':
                        data.processedData[key] = (data.processedData[key].astype('single') * \
                                                   data.processedData['mask'][:, :, None]).astype('uint8')
                    else:

                        data.processedData[key] = (data.processedData[key].astype('single') * \
                                                   data.processedData['mask']).astype('uint8')
        logging.info('Mask Frame created')

    def computeMaskFrame(self, data):
        """
        Computes a mask for the present frame using the uint8 version of the struct, dop, and ret frames.
        If PS is not being processed, it will use only the struct file

        Args:
            data (obj):  data storage object
        Output:
            data.processedData['mask']
        """
        if self.processStates['dop'] and self.processStates['struct']:
            data.processedData['mask'] = ((data.processedData['dop'] / 255) >
                                          data.hsvSettings['maskThresholds'][0] / 255) * \
                                         ((data.processedData['struct'] / 255) >
                                          data.hsvSettings['maskThresholds'][1] / 255) * \
                                         ((data.processedData['ret'] / 255) >
                                          data.hsvSettings['maskThresholds'][2] / 255)
        elif 'struct' in self.procState:
            data.processedData['mask'] = ((data.processedData['struct'] / 255) >
                                          data.hsvSettings['maskThresholds'][1] / 255)
        else:
            data.processedData['mask'] = np.ones_like(data.processedData['angio'])

    def computeWeightFrame(self, data):
        """
        Computes a weight frame for the present frame using the uint8 version of the struct, dop, and ret frames.
        If PS is not being processed, it will use only the struct file

        Args:
            data (obj):  data storage object
        Output:
            w (array) : weight frame
        """
        if self.processStates['dop'] and self.processStates['struct']:
            w = cp.clip((data.processedData['dop'] - data.hsvSettings['dopWeight'][0]) /
                        (data.hsvSettings['dopWeight'][1] - data.hsvSettings['dopWeight'][0]),
                        a_min=0, a_max=1)
            w = w * cp.clip((data.processedData['ret']- data.hsvSettings['retWeight'][0]) /
                            (data.hsvSettings['retWeight'][1] - data.hsvSettings['retWeight'][0]),
                            a_min=0, a_max=1)
            w = w * cp.clip((data.processedData['struct'] -
                             data.hsvSettings['structWeight'][0]) /
                            (data.hsvSettings['structWeight'][1] - data.hsvSettings['structWeight'][0]),
                            a_min=0, a_max=1)

        elif self.processStates['struct']:
            w = cp.clip((data.processedData['struct'] - data.hsvSettings['structWeight'][0]) /
                        (data.hsvSettings['structWeight'][1] - data.hsvSettings['structWeight'][0]),
                        a_min=0, a_max=1)
        else:
            w = cp.ones_like(data.processedData['struct'])

        data.processedData['cweight'] = w

    def convertFrameCircularCartesian(self, data):
        """
        Converts all frames to cartesian coordinates for endoscopic, rotational data
        Args:
            data (object) : Support and raw data object
        """
        if self.firstRun:
            for key, val in self.processStates.items():
                if val:
                    if 'tomch1' in key or 'tomch2' in key:
                        pass
                    else:
                        self.rotCartesianIndex[key] = None

        for key, val in self.processStates.items():
            if val:
                if 'tomch1' in key or 'tomch2' in key:
                    pass
                else:
                    data.processedData[key], self.rotCartesianIndex[key] = rotCartesianInterp(data.processedData[key],
                                                                                              self.rotCartesianIndex[
                                                                                                  key])

    def characterizeSystemBG(self, data):
        """Calculate input polarization specific BG using inter-aline median"""

        logging.info('====================================================')
        logging.info('Characterizing system background:')
        if data.startFrame == data.endFrame:
            frames = data.startFrame
            # START CRAWLING THROUGH SINGLE FRAME
            logging.info('Measuring background on frame: {}'.format(frames))
            frame = frames
            logging.info('BG Calibration, loading frame: {}'.format(frame))
            data.loadFringe(frame=frame)
            data.bg1 = cp.zeros((data.ch1.shape[0],2))
            data.bg2 = cp.zeros((data.ch1.shape[0],2))
            data.bg1[:,0] = cp.median(data.ch1[:, 0::2], axis=1)
            data.bg1[:,1]  = cp.median(data.ch1[:, 1::2], axis=1)
            data.bg2[:,0]  = cp.median(data.ch2[:, 0::2], axis=1)
            data.bg2[:,1]  = cp.median(data.ch2[:, 1::2], axis=1)
            c=1
        else:
            if isinstance(self.processOptions['nFramesBGCorr'], list):
                frames = self.processOptions['nFramesBGCorr']
                frames = [x for x in frames if x > data.startFrame and x <  data.endFrame]
                if frames:
                    pass
                else:
                    frames = [data.startFrame]
            else:
                frames = np.arange(data.startFrame - 1, data.endFrame,
                                np.int(np.ceil(data.endFrame / self.processOptions['nFramesBGCorr']))) + 1
            # START CRAWLING THROUGH INTERVALLED FRAMES
            logging.info('Correcting for system BG using Frames: {}'.format(frames))
            c = 0
            data.bg1 = cp.zeros_like(cp.asarray(data.bg1))
            data.bg2 = cp.zeros_like(cp.asarray(data.bg2))
            for frame in frames:
                logging.info('BG Calibration, loading frame: {}'.format(frame))
                data.loadFringe(frame=frame)
                data.bg1[:,0] = data.bg1[:,0] + cp.median(data.ch1[:, 0::2], axis=1)
                data.bg1[:,1] = data.bg1[:,1] + cp.median(data.ch1[:, 1::2], axis=1)
                data.bg2[:,0] = data.bg2[:,0]+ cp.median(data.ch2[:, 0::2], axis=1)
                data.bg2[:,1] = data.bg2[:,1] + cp.median(data.ch2[:, 1::2], axis=1)
                c = c + 1
            data.bg1 = data.bg1 / c
            data.bg2 = data.bg2 / c
            data.reconstructionSettings['bgRemoval'] == 'ofb'

        # Reset the processing routine to first frame
        logging.info('BG Characterized successfuly with {} frames.'.format(c))
        logging.info('Going back to frame: {}'.format(data.startFrame))
        data.loadFringe(frame=data.startFrame)

        logging.info('====================================================')

    def characterizeSystemPS(self, data):
        """Calculate correction matrices for PS processing"""

        logging.info('====================================================')
        logging.info('Characterizing system PS:')
        if data.startFrame == data.endFrame:
            frame = data.startFrame
            logging.info('Characterizing system PS using Frames: {}'.format(frame))
            logging.info('PS Calibration, loading frame: {}'.format(frame))
            data.loadFringe(frame=frame)
            self.tom.reconstruct(data=data)
            correctionMatrix = cp.zeros((3 ,3, self.tom.processedData['sv1'].shape[-1], 1))
            correctionArray = getCorrectionArray(self.tom.processedData['sv1'], self.tom.processedData['sv2'])
            correctionMatrix[:, :, :, 0] = correctionArray
            c = 1
        else:
            if isinstance(self.processOptions['nFramesOACorr'], list):
                frames = self.processOptions['nFramesOACorr']
                frames = [x for x in frames if x > data.startFrame and x <  data.endFrame]
                if frames:
                    pass
                else:
                    frames = [data.startFrame]
            else:
                frames = np.arange(data.startFrame - 1, data.endFrame,
                                np.int(np.ceil(data.endFrame / self.processOptions['nFramesOACorr']))) + 1
            # START CRAWLING THROUGH INTERVALLED FRAMES
            logging.info('Characterizing system PS using Frames: {}'.format(frames))
            logging.info('PS Calibration, loading frame: {}'.format(frames[0]))
            data.loadFringe(frame=frames[0])
            # Need SV1 shape to create correction matrix so doing this outside loop
            self.tom.reconstruct(data=data)
            correctionMatrix = cp.zeros((3, 3, self.tom.processedData['sv1'].shape[-1], len(frames)))
            correctionArray = getCorrectionArray(self.tom.processedData['sv1'], self.tom.processedData['sv2'])
            correctionMatrix[:, :, :, 0] = correctionArray
            c = 1
            for frame in frames[1:]:
                logging.info('PS Calibration, loading frame: {}'.format(frame))
                data.loadFringe(frame=frame)
                self.tom.reconstruct(data=data)
                correctionArray = getCorrectionArray(self.tom.processedData['sv1'], self.tom.processedData['sv2'])
                correctionMatrix[:, :, :, c] = correctionArray
                c = c + 1

        data.psCorrections['symmetry'], data.psCorrections['bins'] = decomposeCorrection(correctionMatrix)
        data.psCorrections['numBins'] = data.psCorrections['bins'].shape[2]
        logging.info('System Characterized successfuly with {} frames.'.format(c))
        data.generatePSCorrections()
        logging.info('PS corrections file generated')
        data.loadFringe(frame=data.startFrame)
        self.tom.reconstruct(data=data)
        logging.info('Reset to frame: {}'.format(data.startFrame))
        logging.info('====================================================')

    def initialize(self, data, procState=None, writeState=None):
        """
        Initialization handler the tomo+struct+angio portion of the reconstruction with desired setting

        Note:
            Mainly allows for recalculation after nZpixels or zoom changes
        Args:
            data (object) : Support and raw data holder
        """
        if not (procState is None):
            self.procState = procState
        if not (writeState is None):
            self.writeState = writeState

        # Keep track of whether this is an instance where writing occurred and use dict
        if self.writeState:
            self.processOptions['writeProcessed'] = 1
            data.processOptions['writeProcessed'] = 1

        self.setState()
        data.reconstructionSettings['processState'] = self.procState

        if data.processOptions['computeBackground']:
            self.characterizeSystemBG(data)

        # Initialize all processing states
        self.tom = Tomogram(mode=data.reconstructionMode['tom'])

        if self.processStates['struct']:
            self.str = Structure(mode=data.reconstructionMode['struct'])

        if self.processStates['angio']:
            self.ang = Angiography(mode=data.reconstructionMode['angio'])

        if self.processStates['dop'] or self.processStates['ret'] or self.processStates['oa']\
                or self.processStates['theta']:
            self.ps = Polarization(mode=data.reconstructionMode['ps'])

        if self.processStates['hsv']:
            self.color = Colormap()
            # self.initializeHSV(data)

        logging.info('====================================================')
        logging.info('Post processer initialized:')
        logging.info('Processing options:')
        for i in self.processOptions.keys():
            logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=data.processOptions[i]))
        logging.info('====================================================')

    # def initializeHSV(self, data, thetaRef=0, hueCCW=0, opacity=0.01, hsvCrop=[0, 0],
    #                   dopWeight=[20, 130], structWeight=[30, 100], retWeight=[10, 100], maskThresholds=[130, 30, 30],
    #                   overwriteSettings=False):
    #     """
    #     Initialize the HSV frame settings
    #     Notes:
    #         This is a predicated function that should be updated to new syles
    #     Args:
    #         data (object) : Support and raw data holder
    #         thetaRef (int) : N/A
    #         hueCCW (int) : N/A
    #         opacity (array) : Opacity value for projections
    #         hsvCrop (int) : Crop area in Z
    #         dopWeight (int) : V weights in H S V
    #         structWeight (int) : V weights in H S V
    #         retWeight (int) : V weights in H S V
    #         maskThresholds (int) : mask thresholds for each DOP,Struct,Ret, in that order
    #     """
    #     if data.hsvSettings['fileInitialized'] is not True or overwriteSettings:
    #         data.hsvSettings['thetaRef'] = thetaRef
    #         data.hsvSettings['hueCCW'] = hueCCW
    #         data.hsvSettings['opacity'] = opacity
    #         data.hsvSettings['hsvCrop'] = hsvCrop
    #         if data.hsvSettings['hsvCrop'][1] == 0 or \
    #                 data.hsvSettings['hsvCrop'][1] >= data.reconstructionSettings['numZOut']:
    #             data.hsvSettings['hsvCrop'][0] = 0
    #             data.hsvSettings['hsvCrop'][1] = data.reconstructionSettings['numZOut']
    #         data.hsvSettings['dopWeight'] = dopWeight
    #         data.hsvSettings['structWeight'] = structWeight
    #         data.hsvSettings['retWeight'] = retWeight
    #         data.hsvSettings['maskThresholds'] = maskThresholds
    #
    #     else:
    #         if data.hsvSettings['hsvCrop'][1] == 0 or \
    #                 data.hsvSettings['hsvCrop'][1] >= data.reconstructionSettings['numZOut']:
    #             data.hsvSettings['hsvCrop'][0] = 0
    #             data.hsvSettings['hsvCrop'][1] = data.reconstructionSettings['numZOut']
    #
    #     self.hsvString = '[hsv[{} {} {} {} {}]p[{} {} {}]'.format(data.hsvSettings['hueCCW'],
    #                                                               data.hsvSettings['thetaRef'],
    #                                                               data.hsvSettings['dopWeight'],
    #                                                               data.hsvSettings['structWeight'],
    #                                                               data.hsvSettings['retWeight'],
    #                                                               data.hsvSettings['maskThresholds'],
    #                                                               data.hsvSettings['hsvCrop'],
    #                                                               data.hsvSettings['opacity'])
    #
    #
    #     logging.info('====================================================')
    #     logging.info('HSV settings initialized:')
    #     for i in data.hsvSettings.keys():
    #         logging.info('Key_Name:"{kn}", Key_Value:"{kv}"'.format(kn=i, kv=data.hsvSettings[i]))
    #     logging.info('====================================================')

    def oopAverageStokes(self, data):
        """
        Manage out of plane averaging of stokes vectors if OOP requested
        Notes:
            Acts on data.processedData['sv1'] and data.processedData['sv2']
        Args:
            data (object) : Support and raw data object
        """
        if self.tom.settings['holdOnGPURam']:
            tp = cp
        else:
            tp = np

        if self.frameCount - 1 == 0:
            self.div = 0
            self.SV1Temp = tp.zeros(self.tom.processedData['sv1'].shape +
                                    (data.psSettings['oopFilter'],), dtype='float32')
            self.SV2Temp = tp.zeros(self.tom.processedData['sv2'].shape +
                                    (data.psSettings['oopFilter'],), dtype='float32')
            sv1 = data.processedData['sv1']
            sv2 = data.processedData['sv2']
            logging.info('Averaging {} out of plane frames'.format(data.psSettings['oopFilter']))
        else:
            if self.frameCount <= data.psSettings['oopFilter']:
                self.div = self.div + 1
                self.SV1Temp[:, :, :, :, self.frameCount - 1] = data.processedData['sv1']
                self.SV2Temp[:, :, :, :, self.frameCount - 1] = data.processedData['sv2']
            else:
                self.div = data.psSettings['oopFilter']
                self.SV1Temp = tp.roll(self.SV1Temp, -1, axis=4)
                self.SV2Temp = tp.roll(self.SV2Temp, -1, axis=4)
                self.SV1Temp[:, :, :, :, -1] = data.processedData['sv1']
                self.SV2Temp[:, :, :, :, -1] = data.processedData['sv2']
            sv1 = tp.sum(self.SV1Temp, axis=4) / self.div
            sv2 = tp.sum(self.SV2Temp, axis=4) / self.div
        return sv1, sv2