# featureExtractor.py

import sys
import util
import numpy as np
import display


class BaseFeatureExtractor(object):
    def __init__(self):
        pass
    
    def fit(self, trainingData):
        """
        Train feature extractor given the training Data
        :param trainingData: in numpy format
        :return:
        """
        pass
    
    def extract(self, data):
        """
        Extract the feature of data
        :param data: in numpy format
        :return: features, in numpy format and len(features)==len(data)
        """
        pass
    
    def visualize(self, data):
        pass


class BasicFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Just regard the value of the pixels as features (in 784 dimensions)
    """
    def __init__(self):
        super(BasicFeatureExtractorDigit, self).__init__()

    def fit(self, trainingData):
        pass
    
    def extract(self, data):
        return data
    
    def visualize(self, data):
        # reconstruction and visualize
        display.displayDigit(data, outfile='visualize/original_digits.png')


class PCAFeatureExtractorDigit(BaseFeatureExtractor):
    """
    Principle Component Analysis(PCA)
    """

    def __init__(self, dimension):
        """
        self.weights: weight to learn in PCA, in numpy format and shape=(dimension, 784)
        self.mean: mean of training data, in numpy format

        :param dimension: dimension to reduction
        """
        super(PCAFeatureExtractorDigit, self).__init__()
        self.dimension = dimension
        self.weights = None
        self.mean = None

    def fit(self, trainingData):
        """
        Train PCA given the training Data

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.mean(a, axis): mean value of array elements over a given axis
        np.linalg.svd(X, full_matrices=False): perform SVD decomposition to X
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.

        :param trainingData: in numpy format
        :return:
        """
        "*** YOUR CODE HERE ***"
        self.mean = np.mean(trainingData, axis=0)
        data = trainingData - self.mean
        _, _, VT = np.linalg.svd(data, full_matrices=False)
        self.weights = VT[:self.dimension]

        return np.dot(trainingData, self.weights.T)

    # util.raiseNotDefined()

    def extract(self, data):
        """

        :param data: in numpy format
        :return: features, in numpy format, features.shape = (len(data), self.dimension)
        """
        "*** YOUR CODE HERE ***"
        return np.dot(data - self.mean, self.weights.T)

    # util.raiseNotDefined()

    def reconstruct(self, pcaData):
        """
        Perform reconstruction of data given PCA features

        :param pcaData: in numpy format, features.shape[1] = self.dimension
        :return: originalData, in numpy format, originalData.shape[1] = 784
        """
        "*** YOUR CODE HERE ***"
        assert pcaData.shape[1] == self.dimension
        return self.mean + np.dot(pcaData, self.weights)

    # util.raiseNotDefined()

    def visualize(self, data):
        """
        Visualize data with both PCA and reconstruction
        :param data: in numpy format
        :return:
        """
        # extract features
        pcaData = self.extract(data)
        # reconstruction and visualize
        reconstructImg = self.reconstruct(pcaData)
        display.displayDigit(np.clip(reconstructImg, 0, 1), outfile='visualize/pca_digits.png')
