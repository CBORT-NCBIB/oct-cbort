import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import colorcet as cc

def checkForCupy():
    """
    Master function for checking if cupy exists.
    Notes:
        This function is required for 2 reasons.
            1. To keep code agnostic to numpy and cupy - allows to always use "cp", reducing mirrored code with "np"
            2. Allow machines without an nvidia gpu to be detected automatically.
    """
    package_name = 'cupy'
    spec = importlib.util.find_spec(package_name)
    import numpy as np

    if spec is None:
        import numpy as cp
        from scipy.ndimage.filters import convolve
        gpuAvailable = 0
        # Missing as numpy within numpy, so adding it for agnostic code
        def asnumpy(arr):
            return arr
        cp.asnumpy = asnumpy
        freeMemory = 0
        e='None'
    else:
        # Try just in case cupy install is bad
        try:
            import cupy as cp
            from cupyx.scipy.ndimage.filters import convolve
            import nvidia_smi
            gpuAvailable = 1
            def asnumpy(arr):
                return arr
            np.asnumpy = asnumpy
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            freeMemory = info.free
            e='None'
        except Exception as err:
            import numpy as cp
            from scipy.ndimage.filters import convolve
            gpuAvailable = 0
            # Missing as numpy within numpy, so adding it for agnostic code
            def asnumpy(arr):
                return arr
            cp.asnumpy = asnumpy
            freeMemory = 0
            e = err          
            

    return cp, np, convolve, gpuAvailable, freeMemory, e

def normalizeData(mat):
    """
    Returned data normalized between 0 and 1
    """
    return ((mat - np.min(mat)) / (np.max(mat) - np.min(mat)))

def str2bool(v):
    """
    Converts a string to boolean

    Args:
        v (str) : string to be percieved as true or false
    Returns:
        (bool) : True or false based on string
    """
    return v.lower() in ("yes", "True", "true", "t", "1")

def closestDividableNumber(n, m):
    """
    Finds the second closest perfectly divisable number
    Args:
        n (int): number to be divided
        m (int): number that divides
    Output:
        n2 : 2nd closest multiple
"""
    # Find the quotient
    q = int(n / m)
    # 2nd possible closest number
    n2 = (m * (q + 2))
    return n2

def nextPowerOf2(x):
    """
    Return data to the next power of 2
    Args:
        n (int): number in proximity
    Output:
        n2 : next power of 2
    """
    return 1 if x == 0 else 2 ** np.ceil(np.log2(x))

def sub2ind(array_shape, rows, cols):
    return (rows * array_shape[1] + cols - 1).astype('int')

def rotCartesianInterp(image, polarIndex=None, depth=0, offset=None, polarOffset=0, zoom=1, xyzoom=[0, 0]):
    """
    Rotate rectangular b-scans into cartesian circular bscans for endoscopic data

    Notes:
    Args:
        image (array): image to be transformed
        polarIndex (int):
        depth (int):
        offset (int): Depth offset (Nz)
        polarOffset (int): Polar angular offset in degrees
        zoom (int): Overall zoom
        xyzoom (int): Zoom XY image
    """
    dim = image.shape
    print(dim)
    if polarIndex is None:
        if depth == 0:
            depth = dim[0]
        if offset == None:
            offset = [0, dim[0]]

        deltaD = 2 * np.diff(offset) / depth

        left = -np.diff(offset) / zoom * (1 - 1 / depth) + deltaD * xyzoom[0]
        right = np.diff(offset) / zoom * (1 - 1 / depth) + deltaD * xyzoom[0]
        bottom = -np.diff(offset) / zoom * (1 - 1 / depth) + deltaD * xyzoom[1]
        top = np.diff(offset) / zoom * (1 - 1 / depth) + deltaD * xyzoom[1]

        [xx, yy] = np.meshgrid(np.linspace(left, right, depth), np.linspace(bottom, top, depth))

        zpolar = np.round(np.sqrt(xx ** 2 + yy ** 2) + offset[0])
        mask = zpolar < 1
        zpolar[mask] = 1
        mask = zpolar > min(offset[1], dim[0])
        zpolar[mask] = min(offset[1], dim[0])

        xpolar = np.mod(np.round((np.arctan2(-yy, xx) / 2 / np.pi + polarOffset / 360) * (dim[1])), dim[1]) + 1

        polarIndex = sub2ind([dim[0], dim[1]], zpolar, xpolar)
        polarIndex[mask] = dim[0] * dim[1]
        polarIndex[polarIndex > dim[0] * dim[1] - 1] = dim[0] * dim[1] - 1

    if len(dim) < 3:
        image = image.flatten()
        edge = len(image) - 1
        polarImage = image[polarIndex]
        polarImage[edge] = 0
    else:
        polarImage = []
        for i in range(dim[2]):
            frame = image[:, :, i].flatten()
            edge = len(frame) - 1
            frame[edge] = 0
            temp= frame[polarIndex]
            polarImage.append(temp)
        polarImage = np.asarray(polarImage).transpose(1,2,0)
    return polarImage, polarIndex

def showProcessedData(data, processer, zrange = None, mask = True, thresh = 120):
    """
    Quick and easy way to view all the processed data contrasts.
    Args:
        data (object): Dataset object
        processer (object): Post object
        zrange (array): Z range for plotting ie. [100 900]
        mask (bool): Perform masking
        thresh (int): Mask Threshold on DOP ie. 120
    Notes:
        Excluded frames and colormap for theta contrast is designated
        at the top of the method.

    """
    exclude = 'tomch1+tomch2+sv1+sv2+weight+hsv'
    cmap = cc.cm['CET_C2']

    count = 0
    keys = ''
    ps = 'oa+ret+theta'
    vectorial = 'oa+theta'


    for key, val in processer.processStates.items():
        if val and np.sum(data.processedData[key]) > 1 and key not in exclude:
            if count==0:
                if not (zrange is None):
                    zrange = np.arange(zrange[0], zrange[1])
                else:
                    zrange = np.arange(0, data.processedData[key].shape[0])
            keys = keys + key
            count = count + 1

    if mask == True:
        if 'dop' in keys:
            mask = data.processedData['dop'] > thresh
        else:
            mask = 1

    figure = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    fcount = 1
    for key, val in processer.processStates.items():
        if key in keys:
            if key in ps:
                ax = figure.add_subplot(np.int(np.ceil(count / 2)), 2, fcount)
                plt.title(key)
                if key in vectorial:
                    try:
                        img = ax.imshow((np.real(data.processedData[key][zrange]).astype('single')/255*mask[zrange]),
                                        aspect='auto', cmap=cmap)
                    except:
                        img = ax.imshow((np.real(data.processedData[key][zrange]).astype('single')/255*mask[zrange,:,None]),
                                        aspect='auto', cmap=cmap)
                else:
                    img = ax.imshow((np.real(data.processedData[key][zrange]).astype('single')*mask[zrange]).astype('uint8'),
                                    aspect='auto', cmap='gray')
                plt.colorbar(img, orientation='vertical')
                fcount = fcount + 1
            else:
                ax = figure.add_subplot(np.int(np.ceil(count / 2)), 2, fcount)
                plt.title(key)
                img = ax.imshow(np.real(data.processedData[key][zrange]), aspect='auto', cmap='gray')
                plt.colorbar(img, orientation='vertical')
                fcount = fcount + 1

# def filter1d(image, filt):
#     """
#     Easy 1D conv filter, maximized for efficiency for specific array and filter shape.
#         Jit compatible. Also GPU compatible.
#     """
#     M = image.shape[0]
#     Mf = filt.shape[0]
#     Mf2 = Mf // 2
#     filtr = np.repeat(filt, image.shape[1]).reshape(Mf, image.shape[1])
#     result = np.zeros_like(image)
#     for i in prange(Mf2, M - Mf2):
#         result[i, :] = np.sum(filtr * image[i - Mf2:i + Mf2 + 1, :], axis=0)
#     return result
#
# @njit(parallel=True)
# def fastfilter1dJit(image, filt):
#     """ Same as filter1D, Jit compiled.
#     """
#     M = image.shape[0]
#     Mf = filt.shape[0]
#     Mf2 = Mf // 2
#     filtr = np.repeat(filt, image.shape[1]).reshape(Mf, image.shape[1])
#     result = np.zeros_like(image)
#     for i in prange(Mf2, M - Mf2):
#         result[i, :] = np.sum(filtr * image[i - Mf2:i + Mf2 + 1, :], axis=0)
#     return result
