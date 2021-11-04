from oct.utils import *

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()

"""
angioUtils is a collection of classes and functions that facilitate angiography 
related processing.

This is an advanced feature that is not yet accessible from automatic processing.
#TODO: integrate for automatic processing
"""

def get_all_comb_pairs(M, b_monitor=False):
    """returns all possible combination pairs from M repeated measurements (M choose 2)

    Args:
        M (int): number of measurements per

    Returns:
        indices1, incides2
    """       
    indices1 = np.zeros(int(M*(M-1)/2))
    indices2 = np.zeros(int(M*(M-1)/2))
    qq = 0
    for q0 in range(M):
        dt = q0+1
        for q1 in range(M-q0-1):
            indices1[qq] = q1
            indices2[qq] = q1+dt
            qq += 1
    if b_monitor:
        print("indices1:", indices1)
        print("indices2:", indices2)
    return (indices1, indices2)


def generate_angiolines(indices, M, num_alines):
    """generates angio lines from the incides.

    Args:
        indices (np.array[int]): indices among the all possible pairs
        M (int): number of repeats per voxels
        num_alines (int): number of alines in the segment to be processed

    Returns:
        np.array[int]: indices for a tomogram segment
    """    
    angiolines_in_segment = np.tile(indices[None,:], reps=(num_alines,1))

    angiolines_in_segment = angiolines_in_segment + np.arange(num_alines)[:,None]*M
    return angiolines_in_segment.flatten().astype(int)


class MotionRejectionCDV:
    """A class to store parameters controling motion rejection in CDV.
    Motion rejection is implemented for each segment (determined by N in biSeg scan pattern)
    """    

    def __init__(self, angioSettings, init_dict={}):

        self.settings = {
            'M':angioSettings['imgDepth'],
            'c_int':500,
            'c_cdv':0.3,
            'num_segments':4,
            'num_pairs':5,
        }
        self.angiosettings = angioSettings.copy()
        self.segment = {} #initialize to empty
        self.settings.update(init_dict)
        self.initialize()
        
    def __repr__(self) -> str:
        return "settings:\n"+ str(self.settings) +"\nsegment:\n"+ str(self.segment)
        
    def initialize(self):
        if self.settings['M']>1:
            num_xlocs = int(self.angiosettings['imgWidthAng']/self.settings['num_segments'])
            M = self.settings['M']
            ind1, ind2 = get_all_comb_pairs(M)
            self.segment['AlinesToProcAngioLinesA'] = generate_angiolines(ind1, M, num_xlocs)
            self.segment['AlinesToProcAngioLinesB'] = generate_angiolines(ind2, M, num_xlocs)
            self.segment['imgDepthAng'] = ind1.shape[0]
        else:
            raise ValueError("CDV requires M>1. Check angioSettings.")