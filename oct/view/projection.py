import numpy as np

class Project:
    """
    A projecting object for appending inter-aline projections
    """
    def __init__(self):
        # Projections
        self.crop = []
        # Projection Types
        self.projectionTypes = {
            'max': 0,
            'mean': 0,
            'std': 0,
            'sum': 0}
        # Projection Types
        self.projections = {}

    def populateProjections(self, processer):
        """
        Populates a "projections" dictionary according to the processedData types that are present
        Args:
            processer (object) : reconstruction object that holds the processed images
        Returns:
            self.projections (dict) : empty dictionary of processed data types
        """
        for key, val in processer.processedData.items():
            self.projections[key] = {}

    def populateTypes(self, processer):
        """
        Manages which projections types will be performed
        Args:
            processer (object) : reconstruction object that holds the processed images
        Returns:
            self.projectionTypes (dict) : empty dictionary of processed data types
        """
        for key, val in self.projectionTypes.items():
            if key in processer.processOptions['projType']:
                self.projectionTypes[key] = 1

    def appendBScanProjection(self, image, frameType, data,  mask=None):
        """
        Appends a projected b-scan to the designated projection according to frame type
        Args:
            image (nd.array) : b-scan to be projected
            frameType (str) : Designates the image frame type (ie. 'struct')
        Returns:
            self.projections (dict) : fills in projections
        """
        if mask is None:
            mask = np.ones(image.shape[0:1])

        if frameType == 'hsv':
            projBScan = self.projectHSV(image,
                            mask,
                            opacity=data.hsvSettings['opacity'])
            self.projections[frameType].setdefault('hsv', []).append(projBScan)
            for key, val in self.projectionTypes.items():
                if val:
                    projBScan = self.projectData(image, mask, key)
                    self.projections[frameType].setdefault(key, []).append(projBScan)
        else:
            for key, val in self.projectionTypes.items():
                if val:
                    projBScan = self.projectData(image, mask, key)
                    self.projections[frameType].setdefault(key, []).append(projBScan)

    def projectData(self, image, mask, projType):
        """
        Projects b - scan according to projection type ( ie. 'max')
        Args:
            image (nd.array) : b-scan to be projected
            mask (np.array) : Mask formed from Dop,Ret,Stuct
            projType (str) : Designates the type of projection (ie. 'max')
        Returns:
            projBScan (nd.array) : projected b-scan
        """
        if len(image.shape) > 2:
            image = image * mask[:, :, None]
        else:
            image = image * mask

        if 'max' in projType:
            projBScan = np.max(image, axis=0)
        if 'mean' in projType:
            projBScan = np.mean(image, axis=0)
        if 'std' in projType:
            projBScan = np.std(image, axis=0)
        if 'sum' in projType:
            projBScan = np.sum(image, axis=0)
        return projBScan

    def projectHSV(self, image, mask, opacity):
        """
        Projects b - scan specifically for hsv data, with opacity variable and masking
        Args:
            image (nd.array) : b-scan to be projected
            projType (str) : Designates the type of projection (ie. 'max')
        Returns:
            projBScan (nd.array) : projected b-scan
        """
        image = image/255
        opacity = opacity * mask
        projBScan = np.zeros((image.shape[1],3))
        for i in range(image.shape[0]-1,0,-1):
            projBScan[:, 0] = projBScan[:, 0] * (1-opacity[i, :]) + image[i, :, 0]*opacity[i, :]
            projBScan[:, 1] = projBScan[:, 1] * (1-opacity[i, :]) + image[i, :, 1]*opacity[i, :]
            projBScan[:, 2] = projBScan[:, 2] * (1-opacity[i, :]) + image[i, :, 2]*opacity[i, :]

        return projBScan


