import sys, os
from oct import *
import math
cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()


def testInit(t):

    # Full library test
    success = initialize()
    ch1, ch2 = loadTestData(frame=1)
    procStruct, procAngio, procRet = processTestData(ch1, ch2)

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle('Reconstructed data')
    axs[0].imshow(procStruct[:,::2])
    axs[1].imshow(procAngio[:,::2])
    axs[2].imshow(procRet)
    plt.show()

    print('success = {}'.format(success))

    # Type test
    t2 = t+math.pi
    print('What t2 should be: {}'.format(t2))

    return(t2)

def initialize(dirToMetadata=None,state='struct+angio+ps'):
    global data, tom, struct, angio, ps

    if dirToMetadata is None:
        #initialize to test dataset
        dirToMetadata = "C://Users//SPARC PSOCT//Documents//GitHub//oct-cbort//examples//data//4_BL_Benchtop_Phantom_struct_angio_ps"

    # Initialize objects
    data = Load(directory=dirToMetadata)
    data.reconstructionSettings['processState'] = state
    tom = Tomogram()
    struct = Structure()
    angio = Angiography()
    ps = Polarization()

    return 1

def loadTestData(frame=1):
    global data

    # Load raw channel 1 & 2 test data
    data.loadFringe(frame=frame)
    return cp.asnumpy(data.ch1), cp.asnumpy(data.ch2)

def processTestData(ch1, ch2):
    global data, tom, struct, angio, ps
    data.ch1 = ch1
    data.ch2 = ch2

    # Process
    out_tom = tom.reconstruct(data=data)
    out_struct = struct.reconstruct(tomch1=out_tom['tomch1'],tomch2=out_tom['tomch2'], settings=data.structureSettings)
    out_angio = angio.reconstruct(tomch1=out_tom['tomch1'],tomch2=out_tom['tomch2'], settings=data.angioSettings)
    out_ps = ps.reconstruct(sv1=out_tom['sv1'],sv2=out_tom['sv2'], settings=data.psSettings)

    return out_struct['struct'], out_angio['angio'], out_ps['ret']

if __name__ == "__main__":
    print("Imports success")


# class Rt:
#     def __init__(self, inPath, outPath, procState):
#
#         self.inPath = inPath
#         self.outPath = outPath
#         self.procState
#
#         self.tomMode = 'minimal'
#         self.strMode = 'log'
#         self.angMode = 'cdv'
#         self.psMode = 'rt'
#
#     def initialize(self):
#         # Initialize all processing states
#         self.tom = Tomogram(mode=self.tomMode)
#
#         if 'struct' in self.procState:
#             self.str = Structure(mode=self.strMode)
#
#         if 'angio' in self.procState:
#             self.ang = Angiography(mode=self.angMode)
#
#         if 'ps' in self.procState:
#             self.ps = Polarization(mode=self.psMode)
#
#     def reconstructFrame(self):
#         pass
#     def formatImage(self, image):
#         pass
#     def writeImage(self, image):
#         pass
