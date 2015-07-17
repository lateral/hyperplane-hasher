import re
import numpy as np
import sympy as sp
import random as rd
from functools import reduce

NORMAL_VECTOR_ID = 'hyperplane_normal_vector_%s_%i'
NUM_NORMAL_VECS_ID = 'num_normal_vectors_%s'
CHAMBER_ID = 'chamber_%s_%s'
FVECTOR_ID = 'feature_vector_%s'
FVEC_ID_EX = re.compile(r'feature_vector_([\S]*)')


class HyperplaneHasher():

    def __init__(self, kvstore, name, normal_vectors=None):
        """'name' is a string used for cribbing names of things to be stored
        in the KeyValueStore instance 'kvstore'. 'normal_vectors' is
        either a list of 1-rankal numpy arrays, all of the same rank,
        or else of type None. In the latter case, normal vectors are assumed to
        exist in 'kvstore', and are named NORMAL_VECTOR_ID % ('name', i),
        where i is an integer."""
        self.kvstore = kvstore
        self.name = name
        if normal_vectors is None:
            self.num_normal_vectors = kvstore.get_int(
                NUM_NORMAL_VECS_ID % name)
            self.normal_vectors = [kvstore.get_vector(NORMAL_VECTOR_ID % (name, i))
                                   for i in range(self.num_normal_vectors)]
        else:
            self.normal_vectors = normal_vectors
            self.num_normal_vectors = len(normal_vectors)
        self.rank = len(self.normal_vectors[0])

    def _compute_num_chambers(self):
        """Computes the number of chambers defined by the hyperplanes
        corresponding to the normal vectors."""
        d = self.rank
        n = self.num_normal_vectors
        raw_cfs = sp.binomial_coefficients_list(n)
        cfs = np.array([(-1)**i * raw_cfs[i] for i in range(n + 1)])
        powers = np.array([max(entry, 0)
                           for entry in [d - k for k in range(n + 1)]])
        ys = np.array([-1] * len(powers))
        return (-1)**d * sum(cfs * (ys**powers))

    @classmethod
    def _flip_digit(cls, binary_string, i):
        """Given a string 'binary_string' of length n, each letter of
        which is either '0' or '1', and an integer 0 <= i <= n-1, returns
        the binary_string in which the i-th letter is flipped."""
        for letter in binary_string:
            if letter not in ['0', '1']:
                raise ValueError(
                    """Input string contains characters other than '0' and '1'.""")
        if i > len(binary_string) - 1 or i < 0:
            raise ValueError(
                """Argument 'i' outside range 0 <= i <= len(binary_string) - 1.""")
        else:
            flip_dict = {'0': '1', '1': '0'}
            letters = [letter for letter in binary_string]
            letters[i] = flip_dict[binary_string[i]]
            return ''.join(letters)

    @classmethod
    def _hamming_distance(cls, bstring_1, bstring_2):
        """Given two strings of equal length, composed of only 0s and 1s, computes the
        Hamming Distance between them: the number of places at which they differ."""
        for pair in zip(bstring_1, bstring_2):
            if not set(pair).issubset(set(['0', '1'])):
                raise ValueError(
                    """Input strings contain characters other than '0' and '1'.""")
        if len(bstring_1) != len(bstring_2):
            raise ValueError("""Lengths of input strings disagree.""")
        else:
            total = 0
            for i in range(len(bstring_1)):
                if bstring_1[i] != bstring_2[i]:
                    total += 1
            return total

    def _hamming_distance_i(self, chamber_id, i):
        """Given a chamber_id 'chamber_id' and an integer 0 <= i <= self.rank - 1,
        returns the alphabetically sorted list of all chamber_ids having Hamming Distance
        equal to i from 'chamber_id'."""
        for letter in chamber_id:
            if letter not in ['0', '1']:
                raise ValueError(
                    """Input string contains characters other than '0' and '1'.""")
        if i < 0 or i > self.num_normal_vectors - 1:
            raise ValueError(
                """Argument 'i' outside range 0 <= i <= len(binary_string) - 1.""")
        if len(chamber_id) != self.num_normal_vectors:
            raise ValueError("""len(chamber_id) != self.num_normal_vectors.""")
        else:
            result = []
            cids = self._all_binary_strings()
            for cid in cids:
                if self._hamming_distance(chamber_id, cid) == i:
                    result.append(cid)
            return result

    def _all_binary_strings(self):
        """Returns a list of all binary strings of length
        self.num_normal_vectors."""
        n = self.num_normal_vectors
        strings = [np.binary_repr(i) for i in range(2**n)]
        return ['0' * (n - len(entry)) + entry for entry in strings]

    @classmethod
    def _random_vectors(cls, num, rank):
        """This class method return a list of length 'num' or
        vectors (numpy arrays) of rank 'rank'. Both arguments
        are assumed to be positive integers."""
        vec_list = [
            np.array([rd.random() - 0.5 for i in range(rank)]) for j in range(num)]
        return vec_list

    def label_chamber(self, chamber_id, label):
        """Appends the string 'label' to the set with key
        'chamber_id' in self.kvstore, if such exists. If not, then
        a new singleton set {'label'} is created in self.kvstore
        with key 'chamber_id'. The method is idempotent."""
        full_chamber_id = CHAMBER_ID % (self.name, chamber_id)
        full_label_id = FVECTOR_ID % label
        self.kvstore.add_to_set(full_chamber_id, full_label_id)

    def bulk_label_chamber(self, chamber_ids, labels):
        """The arguments 'chamber_ids' and 'labels' must be lists of strings
        of equal length, else ValueError is raised. This method produces the same result
        as calling self.label_chamber(ch_id, label) for all pairs (ch_id, label) in
        chamber_ids x labels, but may be faster if self.kvstore is an instance of
        class DynamoDBAdapter."""
        chamber_ids = [CHAMBER_ID %
                       (self.name, chamber_id) for chamber_id in chamber_ids]
        labels = [FVECTOR_ID % label for label in labels]
        self.kvstore.bulk_add_to_set(chamber_ids, labels)

    def unlabel_chamber(self, chamber_id, label):
        """Removes 'label' from the set corresponding to 'chamber_id'.
        Raises KeyError if 'label' is not an element of the
        corresponding set."""
        full_chamber_id = CHAMBER_ID % (self.name, chamber_id)
        full_label_id = FVECTOR_ID % label
        self.kvstore.remove_from_set(full_chamber_id, full_label_id)

    def chamber_labels(self, chamber_id):
        """Returns the set of labels corresponding
        to key chamber_id. Returns empty set if
        chamber_id is unknown."""
        try:
            full_chamber_id = CHAMBER_ID % (self.name, chamber_id)
            result = set([FVEC_ID_EX.findall(entry)[0] for entry in self.kvstore.get_set(
                full_chamber_id) if len(FVEC_ID_EX.findall(entry)) > 0])
            return result
        except KeyError:
            return set()

    def get_chamber_id(self, vector):
        """Returns the chamber_id of the chamber to which
        vector belongs. Throws a ValueError if rank(vector) differs
        from the ranks of the normal vectors. The binary digits
        of the chamber_id for vectors are computed in the order
        given by the output of the get_normal_vectors() method."""
        if len(vector) != self.rank:
            raise ValueError("""len(vector) != self.rank""")
        else:
            PMZO = {1: 1, -1: 0}
            signs = [int(np.sign(np.dot(vector, nvec)))
                     for nvec in self.normal_vectors]
            chamber_id = ''.join([str(PMZO[entry]) for entry in signs])
            return chamber_id

    def get_chamber_ids(self):
        """Returns the set of all chamber ids."""
        chamber_id_prefix = 'chamber_%s' % self.name
        chamber_id_ex = re.compile(r'%s_([\S]*)' % chamber_id_prefix)
        chamber_ids = [''.join(chamber_id_ex.findall(entry))
                       for entry in self.kvstore.get_set_ids()]
        return set([entry for entry in chamber_ids if len(entry) > 0])

    def adjacent_chamber_ids(self, chamber_id):
        """Returns the set of ids of all chambers directly adjacent
        to the chamber corresponding to 'chamber_id'."""
        results = set([chamber_id])
        for i in range(len(chamber_id)):
            results.add(self._flip_digit(chamber_id, i))
        results = sorted(results)
        return results

    def proximal_chamber_ids(self, chamber_id, num_labels):
        """This method returns the smallest list of chamber ids proximal to
        the string 'chamber_id', such that the union of the corresponding chambers
        contains at least 'num_labels' labels, assumed to be a positive integer.
        The list is sorted by ascending distance.

        NOTE: A set S of chambers is _proximal_ to a given chamber C if
        (i) C is in S, and (ii) D in S implies all chambers nearer to
        C than D are also in S. Here, the distance between two chambers
        is given by the alphabetical distance of their ids."""
        total = 0
        pids = []
        for i in range(self.num_normal_vectors):
            if total >= num_labels:
                break
            hdi = self._hamming_distance_i(chamber_id, i)
            for j in range(len(hdi)):
                if total >= num_labels:
                    break
                next_id = hdi[j]
                total += len(self.chamber_labels(next_id))
                pids.append(next_id)
                if total >= num_labels:
                    break
        return pids

    def proximal_chamber_labels(self, chamber_id, num_labels):
        """Finds the smallest set of proximal chambers containing
        at least 'num_labels' labels, assumed to be a positive integer,
        and returns the set of all labels from this."""
        pcids = self.proximal_chamber_ids(chamber_id, num_labels)
        labels_list = [self.chamber_labels(cid) for cid in pcids]
        labels = reduce(lambda x, y: x.union(y), labels_list)
        return labels

    def get_normal_vectors(self):
        """Returns the list of normal vectors."""
        return self.normal_vectors
