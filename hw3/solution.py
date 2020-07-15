import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    large_set = [] # a set of consensus set
    matched_pairs_shape = np.array(matched_pairs).shape # a shape of matched_pairs set
    for i in range(0, 10): # run 10 times
        rn = random.randint(0, matched_pairs_shape[0]-1) # generate random number
        selected_pair = matched_pairs[rn] # select random pair
        consensus_set = [] # consensus set
        for j in range(matched_pairs_shape[0]): # run mathed pairs' number times
            other_pair = matched_pairs[j] # select pair
            degree1 = keypoints2[selected_pair[1]][3] - keypoints1[selected_pair[0]][3] # compute degree in pairs
            degree2 = keypoints2[other_pair[1]][3] - keypoints1[other_pair[0]][3]
            diff_degree = abs(degree1 - degree2) # compute difference between two degrees
            if(diff_degree % (2 * math.pi)) > ((orient_agreement  * math.pi / 180) % (2 * math.pi)): # compare orients
                continue

            scale_ratio1 = keypoints2[selected_pair[1]][2] / keypoints1[selected_pair[0]][2] # compute scale of pairs
            scale_ratio2 = keypoints2[other_pair[1]][2] / keypoints1[other_pair[0]][2]
            if (scale_ratio1 * (1 + scale_agreement)) < scale_ratio2 or scale_ratio2 < (scale_ratio1 * (1 - scale_agreement)): # compare scales
                continue

            consensus_set.append(other_pair) # push pair to consensus set
        large_set.append(consensus_set) # push consensus set to large set

    ## find the largest set
    largest_set = np.array(large_set[0])
    for i in range(10):
        selected_set = np.array(large_set[i])
        if largest_set.shape[0] < selected_set.shape[0]:
            largest_set = selected_set

    largest_set = largest_set.tolist()
    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    matched_pairs = [] # a set of matched pairs

    for i in range(0, descriptors1.shape[0]):
        descriptor1 = descriptors1[i]
        angles = np.dot(descriptors2, descriptor1) # dot product
        angles = np.array([math.acos(x) for x in angles]) # inverse cosine
        angles_sorted = np.argsort(angles) # index of sorted angles
        dist = angles[angles_sorted[0]] / angles[angles_sorted[1]] # ratio between smallest angle and second smallest ange

        if dist < threshold: # check distance < threshold
            matched_pairs.append([i, angles_sorted[0]])
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    xy_points_homo = np.array([[xy_points[i][0], xy_points[i][1], 1] for i in range(xy_points.shape[0])]) # regular coordinate to homogeneous coordinate
    xy_points_proj = np.matmul(h, xy_points_homo.T).T # matrix multiply

    ## homogeneous coordinate to regular coordinate
    xy_points_proj_2d = np.array([[x, x] for x in xy_points_proj[:, 2]])
    xy_points_out = np.divide(xy_points_proj[:, 0:2], xy_points_proj_2d + 1e-10)
    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    large_set = np.zeros(shape=num_iter) # a set of consensus sets
    h_set = np.ndarray(shape=(num_iter, 3, 3)) # a set of h matrix
    for i in range(num_iter):
        rn = [random.randint(0, xy_src.shape[0]-1) for i in range(4)] # generate 4 random numbers
        xy_src_homo = np.array([[xy_src[i][0], xy_src[i][1], 1] for i in rn]) # select 4 matched_pairs and transform regular coordinate to homogeneous coordinate
        xy_ref_homo = np.array([[xy_ref[i][0], xy_ref[i][1]] for i in rn])

        ## to find h matrix, make this matrix
        xy_src_homo_trans = np.array([
            [xy_src_homo[0][0], xy_src_homo[0][1], 1, 0, 0, 0, -1 * xy_src_homo[0][0] * xy_ref_homo[0][0], -1 * xy_src_homo[0][1] * xy_ref_homo[0][0]],
            [0, 0, 0, xy_src_homo[0][0], xy_src_homo[0][1], 1, -1 * xy_src_homo[0][0] * xy_ref_homo[0][1], -1 * xy_src_homo[0][1] * xy_ref_homo[0][1]],
            [xy_src_homo[1][0], xy_src_homo[1][1], 1, 0, 0, 0, -1 * xy_src_homo[1][0] * xy_ref_homo[1][0], -1 * xy_src_homo[1][1] * xy_ref_homo[1][0]],
            [0, 0, 0, xy_src_homo[1][0], xy_src_homo[1][1], 1, -1 * xy_src_homo[1][0] * xy_ref_homo[1][1], -1 * xy_src_homo[1][1] * xy_ref_homo[1][1]],
            [xy_src_homo[2][0], xy_src_homo[2][1], 1, 0, 0, 0, -1 * xy_src_homo[2][0] * xy_ref_homo[2][0], -1 * xy_src_homo[2][1] * xy_ref_homo[2][0]],
            [0, 0, 0, xy_src_homo[2][0], xy_src_homo[2][1], 1, -1 * xy_src_homo[2][0] * xy_ref_homo[2][1], -1 * xy_src_homo[2][1] * xy_ref_homo[2][1]],
            [xy_src_homo[3][0], xy_src_homo[3][1], 1, 0, 0, 0, -1 * xy_src_homo[3][0] * xy_ref_homo[3][0], -1 * xy_src_homo[3][1] * xy_ref_homo[3][0]],
            [0, 0, 0, xy_src_homo[3][0], xy_src_homo[3][1], 1, -1 * xy_src_homo[3][0] * xy_ref_homo[3][1], -1 * xy_src_homo[3][1] * xy_ref_homo[3][1]]
        ])

        if np.linalg.det(xy_src_homo_trans) == 0: # if matrix is asingular matrix, don't make invese matrix
            continue

        xy_src_homo_inv = np.linalg.inv(xy_src_homo_trans) # make inverse matrix
        h_tmp = np.matmul(xy_src_homo_inv, xy_ref_homo.reshape(8)) # find h matirx

        ## generate h matrix
        h = np.ndarray(shape=(9))
        h[0:8] = h_tmp
        h[8] = 1


        xy_proj = KeypointProjection(xy_src, h.reshape((3, 3))) # generate projection of src

        # compute distance between projection of src and reference and generate consensus set
        dist = (xy_proj[:, 0] - xy_ref[:, 0])**2 + (xy_proj[:, 1] - xy_ref[:, 1])**2
        consensus_set = [i for i in dist if i <= tol**2]
        consensus_set = np.array(consensus_set)

        large_set[i] = consensus_set.shape[0] # large set stores a number of each consensus set's elements
        h_set[i] = h.reshape(3, 3) # store h to h_set

    largest_set_idx = large_set.argmax() # find largest set's index

    h = h_set[largest_set_idx] # return largest set's h matrix
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
