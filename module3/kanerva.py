import random
import math
import numpy as np
import struct


def float_to_bits(f):
    s = struct.pack('>f', f)
    return hex(struct.unpack('>l', s)[0])

# It is recommended that one read the README before using this class


class SelectiveKanervaCoder:
    def __init__(self, _num_prototypes, _dimensions=2, _eta=.025, _seed=0):
        self.numPrototypes = _num_prototypes
        self.dimensions = _dimensions
        self.eta = _eta
        self.c = int(_num_prototypes * _eta)
        self.seed = _seed if _seed != 0 else np.random.random()
        np.random.seed(_seed)
        self.prototypes = np.random.rand(_num_prototypes, _dimensions)

    def get_features(self, _input):
        d = self.prototypes - _input
        d = np.sqrt(sum(d.T ** 2))  # get Euclidian distance
        indexes = np.argpartition(d, self.c, axis=0)[:self.c]
        phi = np.zeros(self.numPrototypes)
        phi[indexes] = 1
        return phi


class KanervaCoder:
    # alternatively hamming distance could be used
    distanceMeasure = 'euclidian'
    numPrototypes = 50
    dimensions = 1
    threshold = 0.02

    # an alternative to the threshold is to take the X closest points
    numClosest = 10
    prototypes = None
    visitCounts = None
    updatePrototypes = None
    minNumberVisited = 50

    Fuzzy = False

    updateFunc = None

    active_radii = 0
    been_around = False  # set to true once a single prototype has been visited
    # the minNumberVisited

    def __init__(self, _num_prototypes, _dimensions, _distance_measure, _n):
        if _distance_measure != 'hamming' and _distance_measure != 'euclidian':
            raise AssertionError('Unknown distance measure ' + str(
                _distance_measure) + '. Use hamming or euclidian.')
        if _num_prototypes < 0:
            raise AssertionError('Need more than 2 prototypes. ' + str(
                _num_prototypes) + ' given. If 0 given, 50 prototpyes ' +
                                   'are used by default.')

        self.dimensions = _dimensions
        self.distanceMeasure = _distance_measure
        self.numPrototypes = _num_prototypes

        # because each observation is normalized within its range,
        # each prototype can be a random vector where each dimension is within
        # (0-1)
        self.prototypes = np.random.random([self.numPrototypes,
                                            self.dimensions])

        # very much a hack
        self.prototypes[:, -1] = np.around(self.prototypes[:, -1])
        self.prototypes[:, -2] = np.around(self.prototypes[:, -2])

        # this is a counter for each prototype that increases each time a
        # prototype is visited
        self.visitCounts = np.zeros(self.numPrototypes)

        # this is used within the learner to manipulate the prototype location
        self.updatedPrototypes = []

        # this is one thing I have been testing, if we want to manipulate our
        # prototypes (combine/move/add). we should make sure that we have
        # explored the state space sufficinetly enough. minNumberVisited is one
        # way to specify that we want at least one prototype to be visited
        # this number of times before we manipulate our representation
        self.minNumberVisited = self.numPrototypes / 2

        # if updateFunc is 0, perform the representation update function found
        # in the XGame paper
        # if updateFunc is 1, perform the representation update function found
        # in the Case Studies paper
        self.updateFunc = 0

        # the active_radii is defined in the Case Study paper and is used as a
        # radius to find all prototypes that are sufficiently close.
        # setting the active_radii to 0 will include no prototypes, and to 1
        # will include all of them
        self.active_radii = .1

        # This is defined in the Case Study paper as a way of limiting how many
        # prototypes should are activated
        # by a given observation
        self.caseStudyN = _n

        # if false, an array of the indexes of the activated prototypes is
        # returned
        # if true, the distance metric for every prototype is returned
        self.Fuzzy = False

    def compute_hamming(self, data, i):
        """Calculate the Hamming distance between two bit strings"""
        prototype = self.prototypes[i]
        count = 0
        for j in range(self.dimensions):
            z = int(float_to_bits(data[j]), 16) & int(prototype[j], 16)
            while z:
                count += 1
                z &= z - 1  # magic!
        return count

    # The function to get the features for the observation 'data'
    # the argument 'update' is a boolean which indicates whether the
    # representation should check for an update condition (such as meeting the
    # minimal amount of prototype visits).
    # This is useful for debugging
    def __call__(self, data, update=True):
        if self.distanceMeasure == 'euclidian':
            if self.updateFunc == 0:  # XGame Paper

                temp_arr = np.array(
                    [[i, np.linalg.norm(data - self.prototypes[i])] for i in
                     range(len(self.prototypes))])

                near_prototype_indxs = [int(x[0])
                                        for x in sorted(temp_arr,
                                                        key=lambda x: x[1])[
                                                 :self.numClosest]]

                if update:

                    print('Updating XGame')
                    for i in near_prototype_indxs:
                        self.visitCounts[i] += 1

                    if not self.been_around:  # use this so we dont have to
                        # calculated the max every time
                        max_visit = max(self.visitCounts)
                        print('Max visit: ' + str(max_visit))
                        if max_visit > self.minNumberVisited:
                            self.been_around = True

                    if self.been_around:
                        self.update_prototypes_xgame()
                return near_prototype_indxs

            elif self.updateFunc == 1:  # Case Studies

                near_prototype_indxs = []
                data = np.array(data)
                for prototype in range(self.numPrototypes):
                    diff_arr = abs(data - self.prototypes[prototype])
                    u = min([1 - diff / self.active_radii
                             if diff <= self.active_radii
                             else 0
                             for diff in diff_arr])
                    if u > 0:
                        near_prototype_indxs.append(prototype)

                return np.asarray(near_prototype_indxs, dtype=int)

        else:
            # if self.distanceMeasure == 'hamming':
            # fuzzy
            temp_arr = np.array(
                [1 if self.compute_hamming(data, i) < self.threshold else 0
                 for i in range(len(self.prototypes))])

            return np.where(temp_arr == 1)[0]

    # the update algorithm defined in the XGame paper
    def update_prototypes_xgame(self):
        self.updatedPrototypes = []
        most_visited_indices = [i[0]
                                for i in sorted(enumerate(self.visitCounts),
                                                key=lambda x: x[1])]
        count = 0
        for prototype in range(self.numPrototypes):
            if math.exp(-self.visitCounts[prototype]) > random.random():
                self.visitCounts[prototype] = 0
                replacement_prototype_index = most_visited_indices[-(count + 1)]
                self.prototypes[prototype] = self.prototypes[
                    replacement_prototype_index]  # add another prototype

                for dimension in range(self.dimensions):
                    rand_offset = (random.random() - .5) / (
                        self.numPrototypes ^ -self.dimensions)
                    self.prototypes[prototype][
                        dimension] += rand_offset

                self.updatedPrototypes.append(
                    [prototype, self.prototypes[prototype],
                     replacement_prototype_index])
                count += 1

        self.visitCounts = np.zeros(self.numPrototypes)
        self.been_around = False
