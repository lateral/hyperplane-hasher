import threading
import Queue
import numpy as np
import pandas as pd
import regex as re
from nn.key_value_store import *
from ann.hyperplane_hasher import *
from functools import reduce

HH_NAME_i = 'HH_%s_%i'


class HHEnsembleNNLookup(object):

    def __init__(self, rank, name, metric, num_normal_vectors,
                 num_hyperplane_hashers, kvstore):
        """Creates an instance of the HHEnsembleNNLookup object, subject to the
        following parameters:
        (i) 'rank' is the dimension of the space in which the feature vectors live.
        it must be a positive integer.
        (ii) 'name' is a string naming the object, used for cribbing keys of associated data.
        (iii) 'metric' is either 'l1' or 'l2'; a ValueError is raised if this is false.
        (iv) 'num_normal_vecs' is a positive integer defining the number of hyperplanes used in
        each HyperplaneHasher instance.
        (v) 'num_hyperplane_hashers'is a positive integer specifiying the number of
        HyperplaneHasher objects to be created.
        (vi) 'kvstore' in an instance of the class KeyValueStore."""
        if metric not in ['l1', 'l2']:
            raise ValueError("""Argument 'metric' not in set {'l1', 'l2'}.""")
        self.rank = rank
        self.metric = metric
        self.num_normal_vectors = num_normal_vectors
        self.num_hyperplane_hashers = num_hyperplane_hashers
        self.kvstore = kvstore
        self.hhs = [
            HyperplaneHasher(
                name=HH_NAME_i %
                (name,
                 i),
                kvstore=kvstore,
                normal_vectors=HyperplaneHasher._random_vectors(
                    num_normal_vectors,
                    rank)) for i in range(num_hyperplane_hashers)]
        self._populate_chambers()

    def _rank_error(self, vec):
        """Throws a value error if len(vec) != self.rank."""
        if len(vec) != self.rank:
            raise ValueError("""length of 'vec' differs from self.rank.""")

    def _label_chamber_ensemble(self, vec, vec_id):
        """For each HyperplaneHasher object hh in self.hhs, labels the
        appropriate chamber in hh with the string 'vec_id'. Throws ValueError
        if len(vec) != self.rank."""
        self._rank_error(vec)
        for hh in self.hhs:
            chamber_id = hh.get_chamber_id(vec)
            hh.label_chamber(chamber_id, vec_id)

    def _bulk_label_chamber_ensemble(self, vecs, vec_ids):
        """For each HyperplaneHasher object hh in self.hhs, and each pair
        (vec, vec_id) in zip(vecs, vec_ids), labels the appropriate chamber
        in hh with the string 'vec_id'. Throws ValueError if
        len(vec) != self.rank for any vec in vecs, and also if
        len(vecs) != len(vec_ids)"""
        if len(vecs) != len(vec_ids):
            raise ValueError('len(vecs) != len(vec_ids)')
        for vec in vecs:
            self._rank_error(vec)
        for hh in self.hhs:
            chamber_ids = [hh.get_chamber_id(vec) for vec in vecs]
            hh.bulk_label_chamber(chamber_ids, vec_ids)

    def _populate_chambers(self):
        """For each HyperplaneHasher object in self.hhs, adds every feature
        vector in self.kvstore to the appropriate chamber."""
        vec_ids = self.kvstore.get_vector_ids()
        vecs = [self.get_vector(vec_id) for vec_id in vec_ids]
        self._bulk_label_chamber_ensemble(vecs, vec_ids)

    def _get_nn_candidates(self, vec, num_neighbours):
        """Returns a set of vector ids comprising 'num_neighbours'
        approximate nearest neighbours to the vector 'vec'.
        Throws a ValueError if len(vec) != self.rank."""
        self._rank_error(vec)
        candidate_vector_ids = [hh.proximal_chamber_labels(
            hh.get_chamber_id(vec), num_neighbours) for hh in self.hhs]
        return reduce(lambda x, y: x.union(y), candidate_vector_ids)

    def _get_nn_candidates_async(self, vec, num_neighbours):
        def get_labels(hh):
            labels = hh.proximal_chamber_labels(
                hh.get_chamber_id(vec), num_neighbours)
            q.put(labels)
        self._rank_error(vec)
        q = Queue.Queue()
        threads = [
            threading.Thread(target=get_labels, args=(hh,)).start() for hh in self.hhs]
        candidate_vector_ids = [q.get() for hh in self.hhs]
        return reduce(lambda x, y: x.union(y), candidate_vector_ids)

    def get_vector_ids(self):
        """Returns the set of all vector ids from underlying
        KeyValueStore object."""
        return set([FVEC_ID_EX.findall(entry)[0] for entry in self.kvstore.get_vector_ids(
        ) if len(FVEC_ID_EX.findall(entry)[0]) > 0])

    def get_vector(self, vec_id):
        """Returns vector with id 'vec_id'. Throws KeyError
        if this does not exist."""
        full_vec_id = FVECTOR_ID % vec_id
        return self.kvstore.get_vector(full_vec_id)

    def bulk_get_vector(self, vec_ids):
        """Returns a list of vectors with ids in the list 'vec_ids'. The length of
        this list is equal to the number of recognised vec_ids."""
        full_vec_ids = [FVECTOR_ID % vec_id for vec_id in vec_ids]
        return self.kvstore.bulk_get_vector(full_vec_ids)

    def get_rank(self):
        """Returns the rank of the underlying space."""
        return self.rank

    def delete_vector(self, vec, vec_id):
        """Delete vector with the id 'vec_id' from underlying KeyValueStore object.
        Remove also the label 'vec_id' from all chambers associated to all
        HyperplaneHasher objects. Throws KeyError if 'vec_id' is not a key in the
        underlying KeyValueStore object, throws ValueError if len(vec) != self.rank."""
        self._rank_error(vec)
        for hh in self.hhs:
            chamber_id = hh.get_chamber_id(vec)
            hh.unlabel_chamber(chamber_id, vec_id)
        full_vec_id = FVECTOR_ID % vec_id
        self.kvstore.remove_vector(full_vec_id)

    def add_vector(self, vec, vec_id):
        """Add the vector 'vec' with id 'vec_id' to the underlying KeyValueStore
        object, and label the appropriate chamber from each underlying HyperplaneHasher
        object with 'vec_id'. Throws a ValueError if length 'vec' != self.rank."""
        self._rank_error(vec)
        full_vec_id = FVECTOR_ID % vec_id
        self.kvstore.store_vector(full_vec_id, vec)
        self._label_chamber_ensemble(vec, vec_id)

    def bulk_add_vector(self, vecs, vec_ids):
        """For all pairs (vec, vec_id) from vecs x vec_ids, add the vector
        'vec' with id 'vec_id' to the underlying KeyValueStore object, and label the
        appropriate chamber from each underlying HyperplaneHasher
        object with 'vec_id'. Throws a ValueError if length 'vec' != self.rank for any
        vec, or if len(vecs) != len(vec_ids)."""
        if len(vecs) != len(vec_ids):
            raise ValueError('len(vecs) != len(vec_ids)')
        for vec in vecs:
            self._rank_error(vec)
        full_vec_ids = [FVECTOR_ID % vec_id for vec_id in vec_ids]
        self.kvstore.bulk_store_vector(full_vec_ids, vecs)
        self._bulk_label_chamber_ensemble(vecs, vec_ids)

    def find_neighbours(self, vec, num_neighbours):
        """Returns a pandas series whose index comprises the vector ids of the
        approximate 'num_neighbours' to vector 'vec', and whose values are the
        respective distances. It is sorted in ascending order of distance.
        Throws a ValueError if length 'vec' != self.rank."""
        self._rank_error(vec)
        norm_dict = {'l2': None, 'l1': 1}
        vec_ids = self._get_nn_candidates_async(vec, num_neighbours)
        vectors = np.array(self.bulk_get_vector(vec_ids))
        distances = pd.Series(np.linalg.norm(
            vectors - vec, ord=norm_dict[self.metric], axis=1), index=vec_ids)
        result = distances.order()[:num_neighbours]
        return result
