from oct import *
import napari
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider, QFileDialog,QLineEdit
import enum
from napari.layers import Image

class Operation(enum.Enum):
    """A set of valid arithmetic operations for image_arithmetic."""
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide

class MGHView:
    """
    A modified viewer and labeller for .mgh files built on a napari viewer

    Notes:
        Does not require the napari-mgh plugin to be installed as this is a unique napari viewer,
        loading is taken care of within the class, using memory mapping
    """
    def __init__(self, directory, state):
        self.directory = directory
        self.state = state
        self.view = 'BScan'
        self.frame = 1
        self.lPos = 1
        self.efPos = 1
        self.projStates = {
            'struct': 0,
            'angio': 0,
            'weight': 0,
            'dop': 0,
            'ret': 0,
            'theta': 0,
            'hsv':0
            }

    def normalizeData(self, mat):
        """
        Returned data normalized between 0 and 1 using the CPU
        """
        return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
    
    def populateMMap(self, data, viewer):
        for key, val in data.loadStates.items():
            if val:
                try:
                    data.storedData[key] = data.memmapMGH(data.directory, key)
                except:
                    continue

    def updateWindow(self, data, viewer, avg = 1):
        """ Updates the window with variable slicing """
        for key, val in data.loadStates.items():
            if val:
                if self.view == 'BScan':
                    if avg > 1:
                        p1 = np.int(max(self.frame - avg / 2, 0))
                        p2 = np.int(min(self.frame + avg / 2, data.meta[4]-1))
                        r = np.arange(p1, p2)
                        try:
                            viewer.layers[key].data = np.mean(np.asarray(data.storedData[key][:, :, r].T), axis=0)
                        except:
                            viewer.add_image(np.mean(np.asarray(data.storedData[key][:, :, r].T), axis=0), name=key,
                                             blending='opaque')
                    else:
                        try:
                            viewer.layers[key].data = np.asarray(data.storedData[key][:, :, self.frame].T)
                        except:
                            viewer.add_image(np.asarray(data.storedData[key][:, :, self.frame].T), name=key, blending='opaque')

                if self.view == 'LScan':
                    if avg > 1:
                        p1 = np.int(max(self.lPos - avg / 2, 0))
                        p2 = np.int(min(self.lPos + avg / 2, data.meta[2]-1))
                        r = np.arange(p1, p2)
                        try:
                            viewer.layers[key].data = np.mean(np.asarray(data.storedData[key][r, :, :]), axis=0)
                        except:
                            viewer.add_image(np.mean(np.asarray(data.storedData[key][r, :, :]), axis=0), name=key,
                                             blending='opaque')
                    else:
                        try:
                            viewer.layers[key].data = np.asarray(data.storedData[key][self.lPos, :, :])
                        except:
                            viewer.add_image(np.asarray(data.storedData[key][self.lPos, :, :]), name=key,
                                             blending='opaque')
                if self.view == 'EnFace':
                    if avg > 1:
                        p1 = np.int(max(self.efPos - avg / 2, 0))
                        p2 = np.int(min(self.efPos + avg / 2, data.meta[3]-1))
                        r = np.arange(p1, p2)
                        try:
                            viewer.layers[key].data = np.mean(np.asarray(data.storedData[key][:, r, :]), axis=1)
                        except:
                            viewer.add_image(np.mean(np.asarray(data.storedData[key][:, r, :]), name=key,
                                             blending='opaque'), axis=1)
                    else:
                        try:
                            viewer.layers[key].data = np.asarray(data.storedData[key][:, self.efPos, :])
                        except:
                            viewer.add_image(np.asarray(data.storedData[key][:, self.efPos, :]), name=key,
                                             blending='opaque')

        if data.loadStates['theta']:
            try:
                viewer.layers['hsv'].data = self.color.apply(viewer.layers['theta'].data/255,
                                                             weight=viewer.layers['ret'].data/255,
                                                             mask=viewer.layers['dop'].data>180,
                                                             cmap='CET_C2')
            except:
                hsv = self.color.apply(viewer.layers['theta'].data / 255,
                                                        weight=viewer.layers['ret'].data/255,
                                                        mask=viewer.layers['dop'].data > 180,
                                                        cmap='CET_C2')
                viewer.add_image(hsv, name='hsv', blending='opaque')



    def run(self):
        with napari.gui_qt():
            viewer = napari.Viewer()
            data = Processed()
            self.color = Colormap()
            data.state = self.state
            data.directory = self.directory
            data.storedData = {}
            data.setState(state=data.state)
            self.populateMMap(data, viewer)
            self.updateWindow(data, viewer)

            def initStorageDirectory(data):
                data.outputDirectory = os.path.join(data.directory, 'Labelled')
                data.outputDataDirectory = os.path.join(data.outputDirectory, 'Data')
                data.outputLabelDirectory = os.path.join(data.outputDirectory, 'Label')
                if not os.path.exists(data.outputDirectory):
                    os.mkdir(data.outputDirectory)
                if not os.path.exists(data.outputDataDirectory):
                    os.mkdir(data.outputDataDirectory)
                if not os.path.exists(data.outputLabelDirectory):
                    os.mkdir(data.outputLabelDirectory)


            ############# Change axis through frames #############
            @magicgui(auto_call=True,
                      avg={'min': 1, 'max': 1000},
                      frame={"widget_type": "Slider", 'min': 1, 'max': data.meta[4]-1},
                      lPos={"widget_type": "Slider", 'min': 1, 'max': data.meta[2]-1},
                      efPos={"widget_type": "Slider", 'min': 1, 'max': data.meta[3]-1},
                      view={"choices": ['BScan', 'LScan', 'EnFace']},
                      layout={'vertical'})
            def changeAxis(avg: int = 1,
                           frame: int = self.frame,
                           lPos: int = 1,
                           efPos: int = 1,
                           view = 'BScan'):

                if view =='BScan':
                    self.view  = 'BScan'
                    self.frame = frame
                if view =='LScan':
                    self.view = 'LScan'
                    self.lPos = lPos
                if view =='EnFace':
                    self.view = 'EnFace'
                    self.efPos = efPos

                self.updateWindow(data, viewer, avg=avg)

            @magicgui(call_button="Project",
                      EfPos1={'min': 1, 'max': data.meta[3]-1},
                      EfPos2={'min': 1, 'max': data.meta[3]-1},
                      type={"choices": list(self.projStates.keys())},
                      layout={'vertical'})
            def project(type: str='struct',
                        Opacity: float=0.1,
                           EfPos1: int = 1,
                           EfPos2: int = 1):

                r = np.arange(EfPos1, EfPos2)
                if type =='hsv':
                    c=1
                    for i in r:
                        theta = data.storedData['theta'][:, i, :]
                        ret = data.storedData['ret'][:, i, :]
                        dop = data.storedData['dop'][:, i, :]
                        if c==1:
                            proj = self.color.apply(theta / 255,
                                             weight = ret / 255,
                                             mask = dop > 180,
                                             cmap = 'hsv').astype('float32')
                        else:
                            proj = proj+self.color.apply(theta / 255,
                                                    weight=ret / 255,
                                                    mask= dop > 180,
                                                    cmap='hsv').astype('float32')
                        c=c+1
                    proj = ((proj-np.min(proj))/np.max(proj-np.min(proj))*255).astype('uint8')
                else:
                    proj = np.sum(np.asarray(data.storedData[type][:, r, :]), axis=1)
                    proj = ((proj-np.min(proj))/np.max(proj-np.min(proj))*255).astype('uint8')
                viewer.add_image(proj)


            ############# Change load state #############
            @magicgui(call_button="Change load state",
                      layout='vertical')
            def changeState(State: str = data.state):
                """ Update mgh load state """
                data.storedData = {}
                data.state = str(State)
                data.setState(data.state)
                
                for key, val in data.loadStates.items():
                    if val:
                        data.storedData[key] = data.memmapMGH(data.directory, key)

                self.updateWindow(data, viewer)


            @viewer.bind_key('d', overwrite=True)
            def next_label(event=None):
                """Keybinding to advance to the next frame"""

                self.frame = np.int(min(self.frame + 1, data.meta[4]-1))
                print('Viewing frame {}'.format(self.frame))

                self.updateWindow(data, viewer)


            @viewer.bind_key('a', overwrite=True)
            def next_label(event=None):
                """Keybinding to advance to the next frame"""

                self.frame = np.int(min(self.frame - 1, data.meta[4]-1))
                print('Viewing frame {}'.format(self.frame))

                self.updateWindow(data, viewer)

            ############# Segment n images in stack #############
            @magicgui(call_button="Start Segmentation", layout='horizontal')
            def startSegment(DataLayer: Image, ArithmeticLayer: Image, Arithmetic: Operation, Average: int = 5,
                             NumFrames: int = 10):

                data.fileName = 'data_' + str(self.frame) + '.tif'
                data.LayerName = DataLayer.name

                # Create Storage directory for labelled images
                initStorageDirectory(data)

                # Get Range
                self.frame = 0
                startFrame = np.int(max(self.frame - np.int(Average) / 2, 0))
                endFrame = np.int(min(self.frame + np.int(Average) / 2, data.meta[4] - 1))
                frameRange = np.arange(startFrame, endFrame)

                # Show Image and new label
                image = self.normalizeData(np.mean(np.asarray(data.storedData[data.LayerName][:, :, frameRange].T), axis=0).astype(
                    'float32')) * self.normalizeData(np.mean(np.asarray(data.storedData[ArithmeticLayer.name][:, :, frameRange].T),
                                         axis=0).astype('float32'))
                image = image / np.max(image)

                data.dataName = 'data_' + str(self.frame)
                data.labelName = 'label_' + str(self.frame)
                viewer.add_image(image, name=data.dataName, blending='opaque')
                viewer.add_labels(np.zeros_like(image), name=data.labelName, blending='additive')

                @magicgui(call_button="Next Frame" ,layout='horizontal')
                def nextFrame(Ignore:int=1):
                    # Update to save non-averaged
                    viewer.layers[data.dataName].data = np.asarray(data.storedData[data.LayerName][:, :, self.frame].T)

                    viewer.layers[data.dataName].save(os.path.join(data.outputDataDirectory, data.dataName + '.tif'))
                    viewer.layers[data.labelName].save(os.path.join(data.outputLabelDirectory, data.labelName + '.tif'))

                    # Remove old label
                    viewer.layers.remove(data.dataName)

                    # Get Range
                    self.frame = np.int(min(self.frame + np.int(data.meta[4]/NumFrames),data.meta[4])-1)

                    startFrame = np.int(max(self.frame - np.int(Average) / 2, 0))
                    endFrame = np.int(min(self.frame + np.int(Average) / 2, data.meta[4]) - 1)
                    frameRange = np.arange(startFrame, endFrame)

                    # Get Names
                    data.dataName = 'data_' + str(self.frame)
                    data.labelName = 'label_' + str(self.frame)

                    # Show Image and new label
                    image = self.normalizeData(np.mean(np.asarray(data.storedData[data.LayerName][:, :, frameRange].T), axis=0).astype(
                        'float32')) * self.normalizeData(np.mean(np.asarray(data.storedData[ArithmeticLayer.name][:, :, frameRange].T),
                                             axis=0).astype('float32'))
                    image = image / np.max(image)

                    viewer.add_image(image, name=data.dataName, blending='opaque')
                    viewer.add_labels(np.zeros_like(image), name=data.labelName, blending='translucent')


                viewer.window.add_dock_widget(nextFrame.native, area='bottom')


            viewer.window.add_dock_widget(startSegment.native, area='bottom')
            viewer.window.add_dock_widget(changeState.native, area='right', name='changeState')
            viewer.window.add_dock_widget(project.native, area='right', name='project')
            viewer.window.add_dock_widget(changeAxis.native, area='right', name='changeState')

            # viewer.window.add_dock_widget(changeDirectory.native, area='right', name='changeDirectory')
            # gui1 = image_arithmetic.Gui()
            # viewer.window.add_dock_widget(image_arithmetic.native, area='right')
            # viewer.layers.events.changed.connect(lambda x: gui1.refresh_choices())
            # gui2 = makeRgb.Gui()
            # viewer.window.add_dock_widget(makeRgb.native, area='right')
            # viewer.layers.events.changed.connect(lambda x: gui2.refresh_choices())
            # gui3 = scrollFrames.Gui()
            # viewer.window.add_dock_widget(scrollFrames.native, area='right')
            # viewer.layers.events.changed.connect(lambda x: scrollFrames.refresh_choices())
            # gui4 = startSegment.Gui()
            # viewer.layers.events.changed.connect(lambda x: gui4.refresh_choices())
            # gui5 = chooseFrame.Gui()

            # ############# Make Concat frame #############
            # @magicgui(call_button="Make Concatenation")
            # def makeConcat():
            #     """Generates concatenated view for a frame"""
            #     c = 1
            #     rows = 1
            #     for layer in viewer.layers[:]:
            #         key = layer.name
            #         if key in data.storedData:
            #             if np.mod(c, 5) < 1:
            #                 rows = rows + 1
            #             c = c + 1
            #     columns = np.floor(c / rows)
            #     canvas = np.zeros((np.int(data.meta[3] * rows), np.int(data.meta[2] * columns)))
            #     print(rows, columns)
            #     c = 0
            #     r = 0
            #
            #     for layer in viewer.layers[:]:
            #         key = layer.name
            #         if key in data.storedData:
            #             canvas[r * data.meta[3]:(r + 1) * data.meta[3], c * data.meta[2]:(c + 1) * data.meta[2]] = \
            #             viewer.layers[layer.name].data
            #             c = c + 1
            #             if c >= columns:
            #                 c = 0
            #                 r = r + 1
            #
            #     viewer.add_image(canvas, name='concat', blending='opaque')
            # viewer.window.add_dock_widget(makeConcat.native, area='right', name='makeConcat')