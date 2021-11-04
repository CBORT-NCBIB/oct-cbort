from oct import *
#from oct.post import Post
import numpy as np
import napari
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider,QFileDialog,QLineEdit
import enum
from napari.layers import Image

# Enums are a convenient way to get a dropdown menu
class Operation(enum.Enum):
    """A set of valid arithmetic operations for image_arithmetic."""
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


class OFDView:
    """
    A real time viewer for .ofd files built on a napari viewer

    Notes:
        Processing is done on the fly to view different processing settings before processing whole dataset.
    """
    def __init__(self, directory, state):
        self.directory = directory
        self.state = state
        self.loadedData = {}
        self.specialCase = 'tomch1+tomch2+sv1+sv2+k1+k2'

    def quickProjection(self, data, xs, ys, zs):
        xsubsample = data.scanSettings['imgDepth'] * xs
        ysubsample = ys
        zsubsample = zs

        data.mmap2Channel()

        windowLength = np.int(data.reconstructionSettings['numSamples'] / zsubsample)
        alineLength = np.arange(0, data.reconstructionSettings['numSamples'])
        window = np.arange(np.int(data.reconstructionSettings['numSamples'] / 2 - windowLength / 2),
                           np.int(data.reconstructionSettings['numSamples'] / 2 + windowLength / 2))
        fringeWindow = np.hanning(windowLength)[:, None]

        # alines = data.scanSettings['AlinesToProcTomo'][0::data.scanSettings['imgDepth']
        frames = np.arange(0, data.reconstructionSettings['numFrames'], ysubsample).astype('int')

        proj = cp.zeros((np.int(len(data.scanSettings['AlinesToProcTomo']) / xsubsample + 1), len(frames)))

        c = 0
        for frame in frames:
            ch = data.memmap[1, window, :, frame]
            chw = cp.asarray(ch[:, data.scanSettings['AlinesToProcTomo']][:, 0::xsubsample] * fringeWindow)
            tom = cp.fft.fft(chw, axis=0)[0:int(windowLength / 2 - 1), :]
            struct = 10 * cp.log(cp.abs(tom ** 2))
            proj[:, c] = cp.sum(struct, axis=0)
            c = c + 1
        return proj

    def run(self):
        data = Load(directory=self.directory)
        processer = Post()
        processer.processFrameRange(data, startFrame=1, endFrame=1,
                                    procState=self.state,
                                    writeState=False, procAll=False)

        with napari.gui_qt():
            viewer = napari.Viewer()
            for key, val in processer.processStates.items():
                if val:
                    if key in self.specialCase:
                        pass
                    else:
                        viewer.add_image(data.processedData[key], name=key, blending='opaque')

            ############## Generate new edit settings #############
            @magicgui(call_button="Generate Edit Settings")
            def generateEditSettings(Ignore:int=1):
                """Adds, subtracts, multiplies, or divides two image layers of similar shape."""
                data.generateEditSettings()
                print('Generated Edit Settings')
            viewer.window.add_dock_widget(generateEditSettings.native, area='right')

            ############## Generate new edit settings #############
            @magicgui(call_button="Quick Projection",
                      layout="vertical")
            def quickProject(xsub: int = 5, ysub: int = 5, zsub: int = 20):
                """Perform a quick projection of the Structure data"""
                proj = self.quickProjection(data, xsub, ysub, zsub)
                viewer.add_image(cp.asnumpy(proj), scale=[xsub, ysub], name='Projection', blending='opaque')
                print('Generated quick projection')


            ############## Scroll through frames #############
            @magicgui(auto_call=True,
                      frame={"widget_type": "FloatSlider", "max": data.scanSettings['numFrames'], "min": 1})
            def scrollFrame(frame: int = 1):
                frame = int(frame)

                print('Viewing frame: {}'.format(frame))
                processer.processFrameRange(data, startFrame=frame, endFrame=frame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            viewer.window.add_dock_widget(scrollFrame.native, area='bottom', name='scrollFrames')

            ############## Change Process state #############
            @magicgui(call_button="Update Reconstruct",
                      layout='vertical')
            def changeState(state: str = processer.procState):
                processer.procState = str(state)

                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            ############## Change index range #############
            @magicgui(call_button="Change index",
                      layout='vertical',
                      low={'min': 0, 'max':data.reconstructionSettings['numZOut']},
                      high={'min': 0, 'max':data.reconstructionSettings['numZOut']})
            def indexRange(low: int = 0,
                           high: int = 0):

                data.reconstructionSettings['depthIndex'][0] = int(low)
                data.reconstructionSettings['depthIndex'][1] = int(high)

                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')


            ############# Update PS Settings #############
            @magicgui(call_button="Re-initialize PS",
                      layout='vertical',
                      maxRet={'min': 0, 'max': 1000})
            def psSettings(dz: int = data.psSettings['zOffset'],
                           resz: float = data.psSettings['zResolution'],
                           fract: int = data.psSettings['binFract'],
                           xFilter: int = data.psSettings['xFilter'],
                           zFilter: int = data.psSettings['zFilter'],
                           dopThresh: float = data.psSettings['dopThresh'],
                           maxRet: int = data.psSettings['maxRet'],
                           thetaOffset: int = data.psSettings['thetaOffset'],
                           spectralBinning: bool = data.psSettings['spectralBinning'],
                           fastProcessing: bool = data.psSettings['fastProcessing']):


                data.processOptions['spectralBinning'] = bool(spectralBinning)
                data.processOptions['fastProcessing'] = bool(fastProcessing)
                data.psSettings['spectralBinning'] = bool(spectralBinning)
                data.psSettings['fastProcessing'] = bool(fastProcessing)
                data.psSettings['zOffset'] = np.int(dz)
                data.psSettings['zResolution']= np.float(resz)
                data.psSettings['binFract'] = np.int(fract)
                data.psSettings['xFilter'] = np.int(xFilter)
                data.psSettings['zFilter'] = np.int(zFilter)
                data.psSettings['dopThresh'] = np.float(dopThresh)
                data.psSettings['maxRet'] = np.int(maxRet)
                data.psSettings['thetaOffset'] = np.int(thetaOffset)
                data.generateEditSettings()

                print('Viewing frame: {}'.format(data.currentFrame))
                processer.initialize(data, procState=processer.procState, writeState=False)
                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            ############# Update HSV Settings #############
            @magicgui(call_button="Re-initialize HSV",
                      layout='vertical',
                      dopWeightLow={'min': 0, 'max': 255},
                      dopWeightHigh={'min': 0, 'max': 255},
                      retWeightLow={'min': 0, 'max': 255},
                      retWeightHigh={'min': 0, 'max': 255},
                      structWeightLow={'min': 0, 'max': 255},
                      structWeightHigh={'min': 0, 'max': 255},
                      dopThresh={'min': 0, 'max': 255},
                      structThresh={'min': 0, 'max': 255},
                      retThresh={'min': 0, 'max': 255})
            def hsvSettings(opacity: int = data.hsvSettings['opacity'],
                           dopWeightLow: int = data.hsvSettings['dopWeight'][0],
                           dopWeightHigh: int = data.hsvSettings['dopWeight'][1],
                           retWeightLow: int = data.hsvSettings['retWeight'][0],
                           retWeightHigh: int = data.hsvSettings['retWeight'][1],
                           structWeightLow: int = data.hsvSettings['structWeight'][0],
                           structWeightHigh: int = data.hsvSettings['structWeight'][1],
                           dopThresh: int = data.hsvSettings['maskThresholds'][0],
                           structThresh: int = data.hsvSettings['maskThresholds'][1],
                           retThresh: int = data.hsvSettings['maskThresholds'][2]):

                data.hsvSettings['opacity'] = np.float(opacity)
                data.hsvSettings['dopWeight'][0] = np.int(dopWeightLow)
                data.hsvSettings['dopWeight'][1] = np.int(dopWeightHigh)
                data.hsvSettings['retWeight'][0] = np.int(retWeightLow)
                data.hsvSettings['retWeight'][1] = np.int(retWeightHigh)
                data.hsvSettings['structWeight'][0] = np.int(structWeightLow)
                data.hsvSettings['structWeight'][1] = np.int(structWeightHigh)
                data.hsvSettings['maskThresholds'][0] = np.int(dopThresh)
                data.hsvSettings['maskThresholds'][1] = np.int(structThresh)
                data.hsvSettings['maskThresholds'][2] = np.int(retThresh)
                data.generateEditSettings()

                print('Viewing frame: {}'.format(data.currentFrame))
                processer.initialize(data, procState=processer.procState, writeState=False)
                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            ############# Update Structre Settings #############
            @magicgui(call_button="Re-initialize Struct",
                      layout='vertical',
                      contrastLow={'min': -255, 'max': 255},
                      contrastHigh={'min': -255, 'max': 255})
            def structureSettings(invertGray: bool = data.structureSettings['invertGray'],
                            contrastLow: int = data.structureSettings['contrastLowHigh'][0],
                            contrastHigh: int = data.structureSettings['contrastLowHigh'][1]):

                data.structureSettings['invertGray'] = np.bool(invertGray)
                data.structureSettings['contrastLowHigh'][0] = np.int(contrastLow)
                data.structureSettings['contrastLowHigh'][1] = np.int(contrastHigh)

                data.generateEditSettings()

                print('Viewing frame: {}'.format(data.currentFrame))
                processer.initialize(data, procState=processer.procState, writeState=False)
                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            ############# Update Angio Settings #############
            @magicgui(call_button="Re-initialize Angio",
                      layout='vertical',
                      contrastLow={'min': -255, 'max': 255},
                      contrastHigh={'min': -255, 'max': 255})
            def angioSettings(invertGray: bool = data.angioSettings['invertGray'],
                              contrastLow: float = data.angioSettings['contrastLowHigh'][0],
                              contrastHigh: int = data.angioSettings['contrastLowHigh'][1]):
                # xFilter: int = data.angioSettings['xFilter'],
                # yFilter: int = data.angioSettings['yFilter']

                data.angioSettings['invertGray'] = np.bool(invertGray)
                data.angioSettings['contrastLowHigh'][0] = np.int(contrastLow)
                data.angioSettings['contrastLowHigh'][1] = np.int(contrastHigh)
                # data.angioSettings['xFilter'] = np.int(xFilter)
                # data.angioSettings['yFilter'] = np.int(yFilter)

                data.generateEditSettings()

                print('Viewing frame: {}'.format(data.currentFrame))
                processer.initialize(data, procState=processer.procState, writeState=False)
                # processer.ang.initialize(data=data, settings=None, filterSize=(data.angioSettings['xFilter'], data.angioSettings['yFilter']))
                processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
                                            procState=processer.procState,
                                            writeState=False, procAll=False)

                for key, val in processer.processStates.items():
                    if val:
                        if key in self.specialCase:
                            pass
                        else:
                            try:
                                viewer.layers[key].data = data.processedData[key]
                            except:
                                viewer.add_image(data.processedData[key], name=key, blending='opaque')

            viewer.window.add_dock_widget(indexRange.native, area='left')
            viewer.window.add_dock_widget(changeState.native, area='right')
            viewer.window.add_dock_widget(quickProject.native, area='bottom')
            viewer.window.add_dock_widget(angioSettings.native, area='bottom')
            viewer.window.add_dock_widget(structureSettings.native, area='bottom')
            viewer.window.add_dock_widget(psSettings.native, area='right')
            viewer.window.add_dock_widget(hsvSettings.native, area='right')

            #
            # # ############# Perfrom arithmetic on frames #############
            # @magicgui(call_button="execute")
            # def image_arithmetic(layerA: Image, operation: Operation, layerB: Image) -> Image:
            #     """Adds, subtracts, multiplies, or divides two image layers of similar shape."""
            #     intermediate = operation.value(layerA.data, layerB.data)
            #     return intermediate / np.max(intermediate) * 255
            #
            # @magicgui(call_button="Make RGB")
            # def makeRgb(layerR: Image, layerG: Image, layerB: Image) -> Image:
            #     """Adds, subtracts, multiplies, or divides two image layers of similar shape."""
            #     rgb = np.dstack((layerR.data, layerG.data, layerB.data))
            #     return rgb / np.max(rgb)

            # ############# Average a section #############
            # @magicgui(call_button="Average section",
            #           state={'widget_type': QLineEdit},
            #           startFrame={'widget_type': QLineEdit, 'minimum': 1, 'maximum': data.scanSettings['numFrames']},
            #           endFrame={'widget_type': QLineEdit, 'minimum': 1, 'maximum': data.scanSettings['numFrames']})
            # def averageSection(state: str, startFrame: int = 1, endFrame: int = 10):
            #     processer.procState = str(state)
            #     processer.setState()
            #     processer.initialize(data)
            #     frameCount = 1
            #     data.startFrame = int(startFrame)
            #     data.endFrame = int(endFrame)
            #     data.numFramesToProc = int(data.endFrame - data.startFrame + 1)
            #     data.frames = np.arange(data.startFrame, data.endFrame + 1)
            #     storedData = {}
            #     for key, val in processer.processStates.items():
            #         if val:
            #             if 'tomch1' in key or 'tomch2' in key:
            #                 pass
            #             else:
            #                 storedData[key] = np.zeros(
            #                     (data.reconstructionSettings['nZPixels'], data.scanSettings['imgWidth'], data.numFramesToProc),
            #                     dtype='uint8')
            #
            #     if data.startFrame and data.endFrame <= data.numFramesInFile:
            #         for frame in data.frames:
            #             t = time.time()
            #             data.currentFrame = frame
            #             data.loadImageData(data.currentFrame)
            #             print('Processing Frame {}'.format(data.currentFrame))
            #             if data.processOptions['process']:
            #                 processer.computationHandler(data)
            #             for key, val in storedData.items():
            #                 storedData[key][:, :, frameCount - 1] = data.processedData[key]
            #             frameCount = frameCount + 1
            #     for key, val in storedData.items():
            #         temp = np.mean(storedData[key][:, :, :], axis=2)
            #         viewer.add_image(temp, name=key + '_' + str(data.startFrame) + '_' + str(data.endFrame) + '_mean')
            #
            # ############# Choose Frame #############
            # @magicgui(call_button="Choose coordinate space",
            #           coord={'coordinates' : ['cartesian', 'polar']})
            # def chooseCoord(coord: str = 'cartesian'):
            #     nameList = ''
            #     for l in viewer.layers[:]:
            #         nameList = nameList + '+' + viewer.layers[l].name
            #
            #     if 'cartesian' in str(coord):
            #         processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
            #                                     procState=processer.procState,
            #                                     writeState=False, procAll=False)
            #
            #     if 'polar' in str(coord):
            #         processer.processFrameRange(data, startFrame=data.currentFrame, endFrame=data.currentFrame,
            #                                     procState=processer.procState,
            #                                     writeState=False, procAll=False, polar=True)
            #
            #     for key, val in processer.processStates.items():
            #         if val:
            #             if 'tomch1' in key or 'tomch2' in key:
            #                 pass
            #             else:
            #                 if key in nameList:
            #                     viewer.layers[key].data = data.processedData[key]
            #                 else:
            #                     viewer.add_image(data.processedData[key], name=key, blending='opaque')
            #     print('Viewing frame: {}'.format(frame))
            #
            # ############# Choose Frame #############
            # @magicgui(call_button="Go to Frame",
            #           frame={'minimum': 1, 'maximum': data.scanSettings['numFrames']-1,
            #                  'orientation': Qt.Vertical})
            # def chooseFrame(frame: int = 1):
            #     nameList = ''
            #     for l in viewer.layers[:]:
            #         nameList = nameList + '+' + viewer.layers[l].name
            #     processer.processFrameRange(data, startFrame=int(frame), endFrame=int(frame),
            #                                 procState=processer.procState,
            #                                 writeState=False, procAll=False)
            #     for key, val in processer.processStates.items():
            #         if val:
            #             if 'tomch1' in key or 'tomch2' in key:
            #                 pass
            #             else:
            #                 if key in nameList:
            #                     viewer.layers[key].data = data.processedData[key]
            #                 else:
            #                     viewer.add_image(data.processedData[key], name=key, blending='opaque')
            #     print('Viewing frame: {}'.format(frame))
            # @magicgui(call_button="Re-limit Struct",
            #           logLimLow={'widget_type': QLineEdit},
            #           logLimHigh={'widget_type': QLineEdit},)
            # def structSettings(logLimLow: float = data.structureSettings['REF_lowhigh'][0],
            #                    logLimHigh: float = data.structureSettings['REF_lowhigh'][1]):
            #     nameList = ''
            #     for l in viewer.layers[:]:
            #         nameList = nameList + '+' + viewer.layers[l].name
            #
            #     print('Re-processing Struct data')
            #     data.structureSettings['REF_lowhigh'] = [np.float(logLimLow), np.float(logLimHigh)]
            #     processer.computeStructureFrame(data)
            #
            #     key = 'struct'
            #     if key in nameList:
            #         viewer.layers[key].data = data.processedData[key]
            #     else:
            #         viewer.add_image(data.processedData[key], name=key,
            #                          blending='opaque')
            # viewer.window.add_dock_widget(image_arithmetic.native, area='right')
            # viewer.layers.events.inserted.connect(image_arithmetic.reset_choices)
            # viewer.layers.events.removed.connect(image_arithmetic.reset_choices)

            # viewer.window.add_dock_widget(makeRgb.native, area='right')
            # viewer.layers.events.inserted.connect(makeRgb.reset_choices)
            # viewer.layers.events.removed.connect(makeRgb.reset_choices)
            # viewer.layers.events.changed.connect(lambda x: gui2.refresh_choices())
            # viewer.window.add_dock_widget(averageSection.Gui(), area='right')
            # viewer.window.add_dock_widget(chooseFrame.Gui(), area='right')
            # viewer.window.add_dock_widget(structSettings.Gui(), area='right')
            # viewer.window.add_dock_widget(chooseCoord.Gui(), area='right')