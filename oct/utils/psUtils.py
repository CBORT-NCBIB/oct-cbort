from oct.utils import *
import math

cp, np, convolve, gpuAvailable, freeMemory, e = checkForCupy()


def stokes(tomch1, tomch2):
    """
    Compute stokes vectors from complex tomograms

    Note:
    Args:
        tomch1 (array): tomogram channel 1
        tomch2 (array): tomogram channel 2
    Output:
        S0 (array) : I Vector
        S1 (array) : Q Vector
        S2 (array) : U Vector
        S3 (array) : V Vector
    """
    S0 = cp.absolute(tomch1) ** 2 + cp.absolute(tomch2) ** 2
    S1 = cp.absolute(tomch1) ** 2 - cp.absolute(tomch2) ** 2
    S2 = 2 * cp.real(tomch1 * cp.conj(tomch2))
    S3 = -2 * cp.imag(tomch1 * cp.conj(tomch2))

    return S0, S1, S2, S3

def unweaveInputPolarizations(S0, S1, S2, S3):
    """
    Unweave the modulated input polarizations from stokes vectors

    Note:

    Args:
        S0 (array) : I Vector
        S1 (array) : Q Vector
        S2 (array) : U Vector
        S3 (array) : V Vector
    Output:
        SV1 (array) : combined IQUV Vector for input polarization 1
        SV2 (array) : combined IQUV Vector for input polarization 2
    """
    SV1 = cp.concatenate(
        (S0[:, 0::2, cp.newaxis],
         S1[:, 0::2, cp.newaxis],
         S2[:, 0::2, cp.newaxis],
         S3[:, 0::2, cp.newaxis]), axis=2)
    SV2 = cp.concatenate(
        (S0[:, 1::2, cp.newaxis],
         S1[:, 1::2, cp.newaxis],
         S2[:, 1::2, cp.newaxis],
         S3[:, 1::2, cp.newaxis]), axis=2)
    return SV1, SV2

def constructOrthonormal(SVN1, SVN2, complete=False):
    """
    Construct orthonormal tripod from the two input polarization states stokes vectors

    Notes:
        If complete==True , SVN3 is also returned
        SVN3 (array): The cross product of SVN1 and SVN2, to complete measurement matrix ([Nz,Alines,4,Bins])
    Args:
        SVN1 (array): Euclidian normalized QUV portion of stokes vector from input state 1 ([Nz,Alines,4,Bins])
        SVN2 (array): Euclidian normalized QUV portion of stokes vector from input state 2 ([Nz,Alines,4,Bins])
    Returns:
        SVN1 (array): Orthonormalized stokes vector from input state 1 ([Nz,Alines,4,Bins])
        SVN2 (array): Orthonormalized stokes vector from input state 2 ([Nz,Alines,4,Bins])
    """

    na = SVN1 + SVN2
    nb = SVN1 - SVN2
    na = na / cp.sqrt((na * na).sum(axis=2))[:, :, None, :]
    nb = nb / cp.sqrt((nb * nb).sum(axis=2))[:, :, None, :]
    SVN1 = (na + nb) / cp.sqrt(2)
    SVN2 = (na - nb) / cp.sqrt(2)
    na, nb = None, None

    if complete:
        SVN3 = cp.cross(SVN1, SVN2, axis=2)
        return SVN1, SVN2, SVN3
    else:
        return SVN1, SVN2


def computeDOP(SV1, SV2, QUV1, QUV2):
    """Compute DOP from filtered and non-filtered data"""
    dop1 = cp.mean(QUV1 / SV1[:, :, 0, :], axis=2)
    dop2 = cp.mean(QUV2 / SV2[:, :, 0, :], axis=2)
    return (dop1 + dop2) / 2, dop1, dop2


def decomposeJonesMatrix(Jmat):
    """ Decompose 2x2 Jones matrix to retardance and diattenuation vectors """
    Jmat = Jmat / cp.sqrt(cp.linalg.det(Jmat))
    q = cp.array([Jmat[0, 0] - Jmat[1, 1], Jmat[1, 0] + Jmat[0, 1], -1j * Jmat[1, 0] + 1j * Jmat[0, 1]]) / 2

    tr = cp.trace(Jmat) / 2

    c = cp.arccosh(tr)
    csin = c / cp.sinh(c)
    if c == 0:
        csin = 1
    f = 2 * q * csin

    rotVector = -cp.imag(f)
    diatVector = cp.real(f)

    return rotVector, diatVector

def makeJonesMatrix(self, retardance, diattenuation=None):
    """
    Jones Matrix of a retarder. Limited to a single retardance vector.

    Notes:
        J = cosh(c)*p_0 + sinhc(c) * SUM_1^3[f_n * p_n]
        where f_n = (d_n - i r_n) / 2
        and p_n are the Pauli basis.

    Args:
        retardance (array): Retardance vector [rQ, rU, rV]
        diattenuation (array): Diattenuation vector [dQ, dU, dV]
    Returns:
        jonesMat (array): Jones Matrix (2, 2)
    """
    if diattenuation is None:
        diattenuation = np.zeros(retardance.shape)

    dim = retardance.shape
    f = (diattenuation - 1j * retardance) / 2
    c = np.sqrt(np.sum(f ** 2, axis=0)).reshape(1, -1)
    sinch = np.sinh(c) / c
    sinch[c == 0] = 1
    jonesMat = np.array([[1], [0], [0], [1]]) * (np.cosh(c)) + sinch * (
            np.array([[1], [0], [0], [-1]]) * f[0, :].reshape(1, -1) +
            np.array([[0], [1], [1], [0]]) * f[1, :].reshape(1, -1) +
            np.array([[0], [1j], [-1j], [0]]) * f[2, :].reshape(1, -1))

    if np.size(retardance) == 3:
        jonesMat = jonesMat.reshape((2, 2))
    else:
        jonesMat = np.squeeze(jonesMat.reshape(2, 2, dim[1], -1))
    return jonesMat


def unwrapCorrection(rotCorr, wrap=math.pi):
    """
    Unwrap correction matrix across spectral bins
    Notes:
        This is mostly important for tracking the vectors, not for the actual correction
    Args:
        rotCorr (array) : Rotation vector array (3,NBins)
    Returns:
        rotCorr (array) : Unwrapped rotation vector array (3,NBins)
    """
    s = cp.mod(cp.cumsum(cp.sqrt(cp.sum(cp.diff(rotCorr, axis=1) ** 2, axis=0)) > wrap), 2)
    ss = cp.insert(s, 0, 0)

    if cp.sum(ss) > rotCorr.shape[-1] / 2:
        ss = 1 - ss

    retCorr = cp.sqrt(cp.sum(rotCorr ** 2, axis=0))
    rotCorr = rotCorr - rotCorr / retCorr[None, :] * ss[None, :] * 2 * math.pi

    # maintain a retardation <pi
    if cp.mean(cp.sqrt(cp.sum(rotCorr ** 2, axis=0))) > math.pi:
        rotCorr = rotCorr - rotCorr / cp.sqrt(cp.sum(rotCorr ** 2, axis=0))[None, :] * 2 * math.pi;

    return rotCorr


def getSymmetryCorrection(MM, dop):
    """
    Obtain a rotation matrix that can be applied to make the measurement matrix symmetric
    Notes:
    Args:
        MM (array): Measurement matrix ([3,3,Nz,Alines,Bins])
        dop (array): Degree of polarization ([3,3,Nz,Alines,Bins])
    Returns:
        retCorr (array): Correction retardation vector ([3, Bins])
    """

    MM[cp.isnan(MM)] = 0
    dop[cp.isnan(dop)] = 0
    dopMask = dop > 0.8

    # First, average the SO3 matrices over the mask ww
    MMProj = cp.zeros((3, 3, MM.shape[-1]))
    for i in range(3):
        for j in range(3):
            MMProj[i, j, :] = dopMask.flatten()[:, None].T @ cp.reshape(MM[i, j, :, :, :], (
            int(dopMask.shape[0] * dopMask.shape[1]), MM.shape[-1]))

    # Create holder correction arrays
    retCorrVector = cp.zeros((3, MMProj.shape[-1]))
    diatCorrVector = cp.zeros((3, MMProj.shape[-1]))

    # Hermitian matrix transform indices
    hInd = cp.array([0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15])

    # Create Mueller Tranformation matrix (Transforms Jones Kroenecker matrix to a Mueller matrix)
    A = cp.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]])
    A = A / cp.sqrt(2)
    # Create Mueller Matrix from SO3
    JJ = cp.zeros((4, 4,))
    JJ[0, 0] = cp.sum(dopMask)
    for window in range(MMProj.shape[-1]):
        # Reassign Mueller Matrix from SO3 Bin
        JJ[1::, 1::] = MMProj[:, :, window]

        # Convert to hermitian symmetric matrix
        muelMat = cp.conj(A.conj().T @ JJ @ A)
        H = (muelMat.flatten()[hInd]).reshape((4, 4))

        # Get eigen vectors of the hermitian matrix
        [eigVal, eigVec] = cp.linalg.eigh(cp.diag([-1, 1, -1, 1]) @ H @ cp.diag([-1, 1, -1, 1]))

        # Get eigenvector with smallest eigenvector
        pos = cp.argmin(abs(eigVal))
        corrMat = eigVec[:, pos]

        # Create Jones correction Matrix
        jonesCorrMatrix = cp.array([[corrMat[1], corrMat[3]], [corrMat[0], corrMat[2]]])
        jonesCorrMatrix = jonesCorrMatrix / cp.sqrt(cp.linalg.det(jonesCorrMatrix))

        # Convert back to stokes, ignoring diatt
        [retVector, diatVector] = decomposeJonesMatrix(jonesCorrMatrix)

        # Collect correction vectors for all spectral bins
        retCorrVector[:, window] = cp.real(retVector)
        diatCorrVector[:, window] = cp.real(diatVector)

        # Error tracking
        jonesVector = cp.array(
            [jonesCorrMatrix[1, 0], jonesCorrMatrix[0, 0], jonesCorrMatrix[1, 1], jonesCorrMatrix[0, 1]])[None, :].T
        errOpt = cp.real(
            jonesVector.T @ cp.diag([-1, 1, -1, 1]) @ H @ cp.diag([-1, 1, -1, 1]) @ (jonesVector / 4 / cp.sum(dopMask)))

        # Error tracking  2: "more pragmatically, a pure V-component matrix M with pi retardation results in an error of 1"
        jonesVector = cp.array([0, 1, 1, 0]).T
        errInit = cp.real(
            jonesVector.T @ cp.diag([-1, 1, -1, 1]) @ H @ cp.diag([-1, 1, -1, 1]) @ (jonesVector / 4 / cp.sum(dopMask)))

    retCorrVector = unwrapCorrection(retCorrVector, wrap=math.pi)
    retCorrMatrix = makeRotationMatrix(retCorrVector)

    return retCorrMatrix, errInit, errOpt


def makeRotationMatrix(rotVector):
    """
    Converts rotation vector into 3x3 rotation matrix
    Notes:

    Args:
        rotVector (array) : Rotation vector array (3,NBins)
    Returns:
        rotMat (array) : Rotation matrix array (3,3,NBins)
    """

    ret = cp.sqrt(cp.sum(rotVector ** 2, axis=0))
    oa = rotVector / ret

    zeroFill = cp.zeros(oa.shape[-1])

    K = cp.asarray([[zeroFill, oa[2, :], - oa[1, :]],
                    [-oa[2, :], zeroFill, oa[0, :]],
                    [oa[1, :], -oa[0, :], zeroFill]])

    KK = cp.asarray([[oa[0, :] ** 2 - 1, oa[0, :] * oa[1, :], oa[0, :] * oa[2, :]],
                     [oa[1, :] * oa[0, :], oa[1, :] ** 2 - 1, oa[1, :] * oa[2, :]],
                     [oa[2, :] * oa[0, :], oa[1, :] * oa[2, :], oa[2, :] ** 2 - 1]])

    rotMatrix = K * cp.sin(ret)[None, None, :] + KK * (1 - cp.cos(ret))[None, None, :] + cp.eye(3)[:, :, None]

    rotMatrix[:, :, ret == 0] = cp.eye(3)[:, :, None]

    return rotMatrix


def decomposeRot(MM, exact=None):
    """
    Decomposes the rotation matrix into the corresponding retardation vector

    Notes:
        If exact=True, normalization of the optic axis is enforced.
        This is usually required if retardation values reach 0 or pi in some locations
    Args:
        MM (array): Measurement matrix ([3,3,Nz,Alines,Bins])
        exact (bool): Flag for normalizing optic axis
    Returns:
        retVector (array): retardation vector
    """
    if (exact is None):
        exact = False

    ret = cp.real(cp.arccos(cp.sum(MM[[0, 1, 2], [0, 1, 2], :, :, :], axis=0) / 2 - 1 / 2))
    if not exact:
        oa = cp.concatenate(((MM[1, 2, :, :, :] - MM[2, 1, :, :, :])[None, :, :, :],
                             (MM[2, 0, :, :, :] - MM[0, 2, :, :, :])[None, :, :, :],
                             (MM[0, 1, :, :, :] - MM[1, 0, :, :, :])[None, :, :, :]), axis=0)
        oa = 1 / 2 / (cp.sin(ret) + 1e-9)[None, :, :, :] * oa
    else:
        # TODO
        oa = cp.concatenate(((MM[1, 2, :, :, :] - MM[2, 1, :, :, :])[None, :, :, :],
                             (MM[2, 0, :, :, :] - MM[0, 2, :, :, :])[None, :, :, :],
                             (MM[0, 1, :, :, :] - MM[1, 0, :, :, :])[None, :, :, :]), axis=0)
        oa = 1 / 2 / (cp.sin(ret) + 1e-9)[None, :, :, :] * oa

    retVector = oa * ret[None, :, :, :]

    retVector[cp.isnan(retVector)] = 0

    return retVector

def applyRotation(MM,rotMatrix):
    """
    Rotates a measurement matrix by a rotation vector
    Notes:
    Args:
        MM (array): Measurement matrix ([3,3,Nz,Alines,Bins])
        rotVector (array): Rotation vector array ([3,3,Bins])
    Returns:
        MM (array): Rotated Measurement matrix ([3,3,Nz,Alines,Bins])
    """
    return cp.einsum('ijmln,jkn->ikmln', MM, rotMatrix)


def getBinCorrection(MM, dop):
    """
    Obtain a rotation matrix that can be applied to align
    the spectral bins of the measurement matrix to reduce PMD

    Notes:
    Args:
        MM (array): Measurement matrix ([3,3,Nz,Alines,Bins])
        dop (array): Degree of polarization ([3,3,Nz,Alines,Bins])
    Returns:
        retCorr (array): Correction retardation vector ([3, Bins])
    """
    dopMask = dop > 0.8
    diagArr = cp.identity(3)
    mid = int(cp.floor(MM.shape[-1] / 2))

    mm = decomposeRot(MM)
    mm[cp.isnan(mm)] = 0
    mm = mm * dopMask[None, :, :, None]
    cmm = mm[:, :, :, mid]

    C = cp.zeros((3, 3, mm.shape[1]))
    retCorr = cp.zeros((3, 3, mm.shape[-1]))

    for window in range(mm.shape[-1]):
        for i in range(mm.shape[0]):
            for j in range(mm.shape[0]):
                C[j, i, :] = cp.sum(mm[i, :, :, window] * cmm[j, :, :], axis=1)
        Cha = C.mean(axis=2)
        [u, s, vh] = cp.linalg.svd(Cha)
        if window == mid:
            vhOut = vh
        diagArr[2, 2] = cp.linalg.det(u @ vh)
        Rha = (u @ diagArr) @ vh
        retCorr[:, :, window] = Rha
    return retCorr, vhOut

def getCorrectionArray(sv1, sv2, stokesFilter=None):
    """
    Compute the system compensating rotation matrix

    Note:

    Args:
        sv1 (array): Stokes vector input polarization 1
        sv2 (array): Stokes vector input polarization 2
        stokesFilter (array): filter for Stokes data
    Output:
    """
    if stokesFilter is None:
        d1 = cp.hamming(15)
        d2 = cp.hamming(5)
        stokesFilter = cp.sqrt(cp.outer(d1, d2))
        stokesFilter = stokesFilter / cp.sum(stokesFilter)

    SV1 = cp.asarray(sv1[:,:,1:4,:].copy())
    SV2 = cp.asarray(sv2[:,:,1:4,:].copy())
    sv1, sv2 = None, None

    mid = np.int(np.ceil(SV1.shape[3] / 2)) - 1
    windowRange = np.arange(0, SV1.shape[3])

    # i^2 = q^2+u^2+v^2 (eq.3.36 Theocaris)>>
    I1 = cp.sqrt((SV1 * SV1).sum(axis=2))
    I2 = cp.sqrt((SV2 * SV2).sum(axis=2))

    # Filter
    SV1 = convolve(SV1, stokesFilter[:, :, None, None], mode='constant')
    SV2 = convolve(SV2, stokesFilter[:, :, None, None], mode='constant')
    I1 = convolve(I1, stokesFilter[:, :, None], mode='constant')
    I2 = convolve(I2, stokesFilter[:, :, None], mode='constant')

    If = cp.mean(I1 * I1 + I2 * I2, axis=-1)

    # Euclidian length of Q,U,V <<after averaging, so different from I1,I2>>
    I1 = (SV1 * SV1).sum(axis=2)
    I2 = (SV2 * SV2).sum(axis=2)
    # Computation of DOP / uniformity

    If = cp.clip(If, a_min=1e-15, a_max=None)
    dop = cp.sqrt(cp.mean(I1 + I2, axis=-1) / If)
    dopMask = (dop > 0.8) * (dop <= 1)

    # Normalize
    SV1 = SV1 / cp.sqrt(I1[:, :, None])
    SV2 = SV2 / cp.sqrt(I2[:, :, None])

    # force the two Stokes vectors to be orthogonal, equivalent to LSQ solution
    SV1 = SV1 + SV2
    SV2 = SV1 - 2 * SV2
    nna = cp.sqrt((SV1 * SV1).sum(axis=2))
    nnb = cp.sqrt((SV2 * SV2).sum(axis=2))
    SV1 = SV1 / nna[:, :, None]
    SV2 = SV2 / nnb[:, :, None]

    I1, I2, If, nna, nnb = None, None, None, None, None

    SV1minus = cp.roll(SV1, -5, axis=0)
    SV1 = cp.roll(SV1, 5, axis=0)
    SV2minus = cp.roll(SV2, -5, axis=0)
    SV2 = cp.roll(SV2, 5, axis=0)

    # Normalized Cross Product
    PA = cp.cross(SV1 - SV1minus, SV2 - SV2minus, axis=2)
    den = cp.sqrt((PA * PA).sum(axis=2))
    den = cp.clip(den, a_min=1e-15, a_max=None)
    PA = PA / den[:, :, None]

    den = None

    # Calculate retsinW
    temp = cp.einsum('ijkl,ijkl->ijl', SV1minus, PA) ** 2
    temp = cp.clip(temp, a_min=1e-15, a_max=1 - .0000001)
    temp2 = (cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1) - temp) / (1 - temp)
    retSinW = cp.arccos(cp.clip(temp2, -1, 1)) / 2 / 5

    temp, temp2 = None, None

    pm = cp.sign((1 - cp.einsum('ijkl,ijkl->ijl', SV1minus, SV1)) *
                 cp.einsum('ijkl,ijkl->ijl', (SV1minus - SV1), (SV2minus + SV2)))

    PA = PA * cp.expand_dims(pm, axis=2)
    PAW = PA * cp.expand_dims(retSinW, axis=2)

    SV2, SV1, SV1minus, SV2minus = None, None, None, None

    PA = cp.nan_to_num(PA)
    ref = PA[:, :, :, mid]

    # Mask DOP
    PA = PA * cp.expand_dims(dopMask[:, :, None], axis=3)
    # Mask edge structures
    edgeMask = cp.ones_like(dopMask)
    edgeMask[0:np.int(PA.shape[0] * 0.2), :] = 0
    edgeMask[-np.int(PA.shape[0]  * 0.2):, :] = 0
    PA = PA * cp.expand_dims(edgeMask[:, :, None], axis=3)

    C = cp.zeros((3, 3, PA.shape[1]))
    correctionArray = cp.zeros((3, 3, PA.shape[-1]))

    for wind in windowRange:
        for i in range(PA.shape[2]):
            for j in range(PA.shape[2]):
                C[j, i, :] = cp.sum(PA[:, :, i, wind] * ref[:, :, j], axis=0)

        correctionArray[:, :, wind] = C.sum(axis=2)

    return correctionArray


def decomposeCorrection(correctionMatrix):
    """
    At the present state, sums the rotational matrices found during PS system characterization.
    Note:
        There are two types of correction matrices found during the SVD. The symCorrection rotational
        matrix is the eigenvectors when A*A-1 on the same bin. The bin correction is the alignment matrix
        between bins.
    Args:
        correctionMatrix (array):  Correction matrix built up over frames
    Output:
        symCorrection (array):
        binCorrection (array):
    """
    binCorrection = cp.zeros((3,3,correctionMatrix.shape[2]))
    symCorrection = cp.zeros((3, 3, 1))

    diagArr = cp.identity(3)
    mid = int(np.ceil(correctionMatrix.shape[2] / 2)) - 1

    for wind in range(correctionMatrix.shape[2]):
        [u, s, vh] = cp.linalg.svd(correctionMatrix[:, :, wind, :].sum(axis=2))
        if wind == mid:
            symCorrection[:, :, 0] = vh
        diagArr[2, 2] = cp.linalg.det(cp.matmul(u, vh))
        Rha = cp.matmul(cp.matmul(u, diagArr), vh)
        binCorrection[:, :, wind] = Rha

    return symCorrection, binCorrection

def filtNormStokes(SV1, SV2, stokesFilter=None):
    """
    Filter and normalize stokes vectors

    Notes:
    Args:
        SV1 (array): Stokes vector from input state 1 ([Nz,Alines,4,Bins])
        SV1 (array): Stokes vector from input state 2 ([Nz,Alines,4,Bins])
    Returns:
        SVF1 (array): Filtered stokes vector from input state 1 ([Nz,Alines,4,Bins])
        SVF2 (array): Filtered stokes vector from input state 2 ([Nz,Alines,4,Bins])
        SVN1 (array): Euclidian normalized QUV portion of stokes vector from input state 1 ([Nz,Alines,4,Bins])
        SVN2 (array): Euclidian normalized QUV portion of stokes vector from input state 2 ([Nz,Alines,4,Bins])
        QUV1 (array): Euclidian length QUV portion of stokes vector from input state 1 ([Nz,Alines,4,Bins])
        QUV2 (array): Euclidian length QUV portion of stokes vector from input state 2 ([Nz,Alines,4,Bins])
    """

    if stokesFilter is None:
        d1 = cp.hamming(11)
        d2 = cp.hamming(3)
        stokesFilter = cp.sqrt(cp.outer(d1, d2))
        stokesFilter = stokesFilter / cp.sum(stokesFilter)

    # Filter
    SVF1 = convolve(SV1, stokesFilter[:, :, None, None], mode='mirror')
    SVF2 = convolve(SV2, stokesFilter[:, :, None, None], mode='mirror')

    # Euclidian length of  stokes vectors
    QUV1 = cp.sqrt((SVF1[:, :, 1:4, :] * SVF1[:, :, 1:4, :]).sum(axis=2))
    QUV2 = cp.sqrt((SVF2[:, :, 1:4, :] * SVF2[:, :, 1:4, :]).sum(axis=2))

    # Normalize QUV information
    SVN1 = SVF1[:, :, 1:4, :] / QUV1[:, :, None, :]
    SVN2 = SVF2[:, :, 1:4, :] / QUV2[:, :, None, :]

    return SVF1, SVF2, SVN1, SVN2, QUV1, QUV2


def makeMeasurementMatrix(SV1, SV2, stokesFilter=None):
    """
    Generate Measurement matrix from raw stokes vectors

    Notes:
    Args:
        SV1 (array): Stokes vector from input state 1, filtered ([Nz,Alines,4,Bins])
        SV2 (array): Stokes vector from input state 1, filtered ([Nz,Alines,4,Bins])
    Returns:
        MM (array): Measurement matrix ([3,3,Nz,Alines,Bins])
        dop (array): Degree of polarization ([Nz,Alines,Bins])
    """

    # Get filtered, normalized filtered, and QUV portions of the stokes vectors
    SVF1, SVF2, SVN1, SVN2, QUV1, QUV2 = filtNormStokes(SV1, SV2, stokesFilter=stokesFilter)
    SV1, SV2 = None, None

    # Compute DOP
    dop, dop1, dop2 = computeDOP(SVF1, SVF2, QUV1, QUV2)
    SVF1, SVF2, QUV1, QUV2 = None, None, None, None

    # Construct orthonormal tripod
    SVN1, SVN2, SVN3 = constructOrthonormal(SVN1, SVN2, complete=True)

    # G enerate Measurement Matrix; dimensions: 3x3,Nz,NAlines,Nw
    MM = cp.concatenate((SVN1[None, :], SVN2[None, :], SVN3[None, :]), axis=0).transpose((0, 3, 1, 2, 4))

    MM[cp.isnan(MM)] = 0
    dop[cp.isnan(dop)] = 0

    return MM, dop