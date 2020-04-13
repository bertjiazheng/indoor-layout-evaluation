import os

from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy

from utils import corner_to_boundary


def compute_pairwise_distances(annotations, predictions):
    """
    Calculates the pairwise distances of junction
    between annotations and predictions.

    Arguments:
        annotations     (ndarray): N x 2
        predictions     (ndarray): M x 2

    Returns:
        distances       (ndarray): N x M
    """
    num_gts = len(annotations)
    num_preds = len(predictions)

    # group the ceiling-wall and floor-wall junctions
    indices = list(range(0, num_gts, 2)) + list(range(1, num_gts, 2))
    annotations = annotations[indices, :]

    indices = list(range(0, num_preds, 2)) + list(range(1, num_preds, 2))
    predictions = predictions[indices, :]

    # compute the pairwise distances (num_gts x num_preds)
    distances = scipy.spatial.distance.cdist(annotations, predictions)

    return distances


def eval_junctions(distances, thresholds=5):
    """
    Calculates precision/recall for junctions between annotations and predictions.

    Arguments:
        distances       (ndarray): N x M
        threshold       (tuple, list)

    Returns:
        F               (float) : junction F-measure
    """
    thresholds = thresholds if isinstance(
        thresholds, tuple) or isinstance(thresholds, list) else thresholds

    num_gts, num_preds = distances.shape

    # filter the matches between ceiling-wall and floor-wall junctions
    mask = np.zeros_like(distances, dtype=np.bool)
    mask[:num_gts//2, :num_preds//2] = True
    mask[num_gts//2:, num_preds//2:] = True
    distances[~mask] = np.inf

    # F-measure under different thresholds
    Fs = []
    for threshold in thresholds:
        distances_temp = distances.copy()

        # filter the mis-matched pairs
        distances_temp[distances_temp > threshold] = np.inf

        # remain the rows and columns that contain non-inf elements
        distances_temp = distances_temp[:, np.any(np.isfinite(distances_temp), axis=0)]

        if np.prod(distances_temp.shape) == 0:
            Fs.append(0)
            continue

        distances_temp = distances_temp[np.any(np.isfinite(distances_temp), axis=1), :]
        
        # solve the bipartite graph matching problem
        row_ind, col_ind = linear_sum_assignment(distances_temp)

        # compute precision and recall
        precision = len(row_ind) / num_preds
        recall = len(col_ind) / num_gts

        # compute F measure
        Fs.append(2 * precision * recall / (precision + recall))

    return Fs


def eval_wireframe(distances, thresholds=5):
    """
    Calculates precision/recall for wireframe between annotations and predictions.

    Arguments:
        distances       (ndarray): N x M
        threshold       (tuple, list)

    Returns:
        F               (float): wireframe F-measure
    """
    thresholds = thresholds if isinstance(
        thresholds, tuple) or isinstance(thresholds, list) else thresholds

    num_gts, num_preds = distances.shape
    # note that the definition of the number is slightly different from eval_junctions,
    # the number here denotes the number of pairs
    num_gts //= 2
    num_preds //= 2

    # initialize the distances between the wireframes (3 * num_gts x 3 * num_preds)
    distances_wireframe = np.full((3 * num_gts, 3 * num_preds), np.inf)

    # compute the pairwise distances between line segments from junction distances
    distances_wireframe[:num_gts, :num_preds] = distances[:num_gts, :num_preds] + \
        np.roll(distances[:num_gts, :num_preds], (-1, -1), axis=(0, 1))
    distances_wireframe[num_gts:2*num_gts, num_preds:2*num_preds] = distances[num_gts:, num_preds:] + \
        np.roll(distances[num_gts:, num_preds:], (-1, -1), axis=(0, 1))
    distances_wireframe[2*num_gts:, 2*num_preds:] = distances[:num_gts, :num_preds] + \
        distances[num_gts:, num_preds:]

    # F-measure under different thresholds
    Fs = []
    for threshold in thresholds:
        distances_temp = distances_wireframe.copy()

        # filter the mis-matched pairs
        distances_temp[distances_temp > threshold] = np.inf

        # remain the rows and columns that contain non-inf elements
        distances_temp = distances_temp[:, np.any(np.isfinite(distances_temp), axis=0)]
        distances_temp = distances_temp[np.any(np.isfinite(distances_temp), axis=1), :]

        if np.prod(distances_temp.shape) == 0:
            Fs.append(0)
            continue

        # solve the bipartite graph matching problem
        row_ind, col_ind = linear_sum_assignment(distances_temp)

        # compute precision and recall
        precision = len(row_ind) / num_preds / 3
        recall = len(col_ind) / num_gts / 3

        # compute F measure
        Fs.append(2 * precision * recall / (precision + recall))

    return Fs


def convert_segmentation(junctions, height=512, width=1024):
    """ convert corner annotations to instance segmentation
    """
    segmentation = np.zeros((height, width), dtype=np.int)

    boundary = corner_to_boundary(junctions, height, width)

    segmentation[np.round(boundary[0]).astype(int), np.arange(width)] = 1
    segmentation[np.round(boundary[1]).astype(int), np.arange(width)] = 1
    segmentation = np.cumsum(segmentation, axis=0)

    horizontal_boundary = np.unique(junctions[::2, 0])

    vertical_wall = np.zeros((height, width), dtype=np.int)
    vertical_wall[:, horizontal_boundary] = 1
    vertical_wall = np.cumsum(vertical_wall, axis=1)
    vertical_wall[vertical_wall == np.max(vertical_wall)] = 0

    segmentation[segmentation != 0] = 3 - segmentation[segmentation != 0]
    segmentation[segmentation == 2] += vertical_wall[segmentation == 2]

    return segmentation


def eval_plane(annotations, predictions, threshold=0.5):
    """
    Calculates precision/recall for planes between annotations and predictions.

    Arguments:
        annotations     (ndarray): N x 2
        predictions     (ndarray): M x 2
        threshold       (float)

    Returns:
        F               (float): planes F-measure
    """
    # generate instance segmentation from junctions
    annotations = convert_segmentation(annotations)
    predictions = convert_segmentation(predictions)

    # get number of planes
    num_gts = len(np.unique(annotations))
    num_preds = len(np.unique(predictions))

    # convert instance segmentation to one hot encoding
    annotations = (np.expand_dims(annotations, -1) ==
                   np.arange(num_gts)).astype(np.bool)
    predictions = (np.expand_dims(predictions, -1) ==
                   np.arange(num_preds)).astype(np.bool)

    # compute intersection over union
    inters = np.sum((np.expand_dims(annotations, -1) &
                     np.expand_dims(predictions, 2)), axis=(0, 1))
    union = np.sum(((np.expand_dims(annotations, -1) |
                     np.expand_dims(predictions, 2)) > 0.5), axis=(0, 1))
    plane_ious = inters / np.maximum(union, 1e-4)

    # matching
    plane_matched = np.sum((plane_ious > threshold).astype(np.float32))

    # compute precision and recall
    precision = plane_matched / num_preds
    recall = plane_matched / num_gts

    # compute F measure
    F = 2 * precision * recall / np.maximum((precision + recall), 1e-6)

    return F


def evaluate(annotations, predictions, thresholds):
    if len(predictions) == 0:
        return {'junction': 0.0, 'wireframe': 0.0, 'plane': 0.0}

    # pre-compute
    distances = compute_pairwise_distances(annotations, predictions)

    F_J = eval_junctions(distances, thresholds['junction'])
    F_W = eval_wireframe(distances, thresholds['wireframe'])
    F_P = eval_plane(annotations, predictions, thresholds['plane'])
    return {'junction': F_J, 'wireframe': F_W, 'plane': F_P}
