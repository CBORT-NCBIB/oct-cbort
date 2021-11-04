from .ofd import *
from .mgh import *
#from .oct import *
import napari
import sys

if __name__ == "__main__":

    #try:
    viewer = sys.argv[1]
    directory = str(sys.argv[2])
    state = str(sys.argv[3])
    print('Viewer chosen: {}'.format(viewer))
    print('Directory chosen: {}'.format(directory))
    print('State chosen: {}'.format(state))

    if 'mgh' in viewer:
        viewer = MGHView(directory, state)
        viewer.run()

    elif 'ofd' in viewer:
        viewer = OFDView(directory, state)
        viewer.run()
    # except:
    #     print('hi')
    #     with napari.gui_qt():
    #         viewer = napari.Viewer()
