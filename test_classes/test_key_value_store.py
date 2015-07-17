from nn.hh_ensemble_lookup import *
from nn.dictionary_store import *
from ann.hyperplane_hasher import HyperplaneHasher, NORMAL_VECTOR_ID, NUM_NORMAL_VECS_ID, CHAMBER_ID
import unittest, copy, string, numpy as np, random as rd, pandas as pd

RANK = 10
NAME = 'test_HHENL'
METRIC = 'l2'
NNV = 5
NHH = 4
NUM_VECS = 30

class TestHHEnsembleLookup(unittest.TestCase):

    def setUp(self):
        """Create a HHEnsembleLookup object whose underlying KeyValueStore object
        is a DictionaryStore instance populated by NUM_VECS feature vectors."""
        self.letters = list(string.ascii_lowercase)
        self.feature_vecs = HyperplaneHasher._random_vectors(NUM_VECS, RANK)
        self.feature_vecs_ids = ['%i' % i for i in range(NUM_VECS)]
        self.hhenl = self._create_hhenl()
        for pair in zip(self.feature_vecs, self.feature_vecs_ids):
            vec, vec_id = pair
            self.hhenl.add_vector(vec, vec_id)

    def _create_hhenl(self):
        """Returns an empty instance of HHEnsembleNNLookup."""
        kvstore = DictionaryStore(dict())
        return HHEnsembleNNLookup(rank=RANK, name=NAME, metric = METRIC, num_normal_vectors=NNV, num_hyperplane_hashers=NHH, kvstore=kvstore)

    def _get_all_hh_labels(self, hh):
        """Returns the set of all labels in all chambers of hh."""
        ch_ids = hh.get_chamber_ids()
        list_set_labels = [hh.chamber_labels(ch_id) for ch_id in ch_ids]
        return reduce(lambda x, y: x.union(y), list_set_labels)

    def _rank_error(self, function):
        """Throws ValueError if len(vec) != self.rank."""
        vec = self.feature_vecs[0]
        vec_id = self.feature_vecs_ids[0]
        self.assertRaises(ValueError, function, *[vec[1:], vec_id])

    def _bulk_rank_error(self, function):
        """Throws ValueError if len(vec) != self.rank for any vec in vecs."""
        vec_short = HyperplaneHasher._random_vectors(1, self.hhenl.rank - 1)
        vecs = HyperplaneHasher._random_vectors(10, self.hhenl.rank) + vec_short
        vec_ids = self.letters[:11]
        self.hhenl._bulk_label_chamber_ensemble(vecs[:10], vec_ids[:10])
        self.assertRaises(ValueError, function, *[vecs, vec_ids])

    def _bulk_list_length_error(self, function):
        """Throws ValueError if len(vec_ids) != len(vec_ids)."""
        vecs = HyperplaneHasher._random_vectors(10, self.hhenl.rank)
        vec_ids = self.letters[:11]
        self.assertRaises(ValueError, function, *[vecs, vec_ids])

    def test_init_1(self):
        """Class attributes are correctly initialised."""
        self.assertEqual(RANK, self.hhenl.rank)
        self.assertEqual(METRIC, self.hhenl.metric)
        self.assertEqual(NNV, self.hhenl.num_normal_vectors)
        self.assertEqual(NHH, self.hhenl.num_hyperplane_hashers)

    def test_init_2(self):
        """Attribute self.hhs is a list of HyperplaneHasher objects of
        length = self.num_hyperplane_hashers. Each HH object has the expected
        value for 'num_normal_vectors'."""
        hhs = self.hhenl.hhs
        self.assertIsInstance(hhs, list)
        for hh in hhs:
            self.assertIsInstance(hh, HyperplaneHasher)
            self.assertEqual(hh.num_normal_vectors, NNV)
        self.assertEqual(len(hhs), self.hhenl.num_hyperplane_hashers)

    def test_init_3(self):
        """Total set of labels in all chambers of any given HyperplaneHasher object
        equals set(self.feature_vecs_ids)."""
        hhs = self.hhenl.hhs
        for hh in hhs:
            chamber_ids = set([hh.get_chamber_id(vec) for vec in self.feature_vecs])
            labels_set_list = [hh.chamber_labels(ch_id) for ch_id in chamber_ids]
            labels_set = reduce(lambda x, y: x.union(y), labels_set_list)
            self.assertEqual(labels_set, set(self.feature_vecs_ids))
            self.assertEqual(len(labels_set), NUM_VECS)

    def test_label_chamber_ensemble_1(self):
        """For each underlying HyperplaneHasher object, a new label is
        added to precisely one chamber. The set of chamber ids present as keys
        in self.kvstore is either unchanged, or enlarged by one element."""
        feature_vecs = self.feature_vecs
        old_chamber_ids = {hh: set([hh.get_chamber_id(vec) for vec in feature_vecs]) for hh in self.hhenl.hhs}
        old_chamber_labels = {hh: [hh.chamber_labels(ch_id) for ch_id in old_chamber_ids[hh]] for hh in self.hhenl.hhs}
        new_vec = HyperplaneHasher._random_vectors(1, self.hhenl.rank)[0]
        self.hhenl._label_chamber_ensemble(new_vec, 'new_vec_id')
        feature_vecs.append(new_vec)
        new_chamber_ids = {hh: set([hh.get_chamber_id(vec) for vec in feature_vecs]) for hh in self.hhenl.hhs}
        new_chamber_labels = {hh: [hh.chamber_labels(ch_id) for ch_id in new_chamber_ids[hh]] for hh in self.hhenl.hhs}
        for hh in self.hhenl.hhs:
            len_diff = len(new_chamber_ids[hh]) - len(old_chamber_ids[hh])
            self.assertIn(len_diff, [0, 1])
            if len_diff == 0:
                #vector 'new_vec' has landed in an existing chamber.
                #the set of chamber ids thus remains unchanged, but
                #exactly one chamber has exactly one new label,
                #namely 'new_vec_id'
                self.assertEqual(old_chamber_ids[hh], new_chamber_ids[hh])
                comparison = list(np.array(old_chamber_labels[hh]) == np.array(new_chamber_labels[hh]))
                expected_bools = set([False] + [True] * (len(old_chamber_ids) - 1))
                self.assertEqual(set(comparison), expected_bools)
                label_diff = new_chamber_labels[hh][comparison.index(False)].difference(old_chamber_labels[hh][comparison.index(False)])
                self.assertEqual(label_diff, set(['new_vec_id']))
            if len_diff == 1:
                #vector 'new_vec' has landed in a new chamber.
                #The id of the new chamber is that of the chamber to
                #which 'new_vec' belongs, and the new chamber
                #is exactly set(['new_vec_id']).
                id_diff = new_chamber_ids[hh].difference(old_chamber_ids[hh])
                self.assertEqual(id_diff, set([hh.get_chamber_id(new_vec)]))
                labels_diff = [entry for entry in new_chamber_labels[hh] if entry not in old_chamber_labels[hh]][0]
                self.assertEqual(labels_diff, set(['new_vec_id']))

    def test_label_chamber_ensemble_2(self):
        """Throws ValueError if len(vec) != self.rank."""
        new_vec_short = HyperplaneHasher._random_vectors(1, self.hhenl.rank - 1)[0]
        self.assertRaises(ValueError, self.hhenl._label_chamber_ensemble, *[new_vec_short, 'new_vec_short_id'])

    def test_bulk_label_chamber_ensemble_1(self):
        """Throws ValueError if len(vec) != self.rank for any vec in vecs."""
        vec_short = HyperplaneHasher._random_vectors(1, self.hhenl.rank - 1)
        vecs = HyperplaneHasher._random_vectors(10, self.hhenl.rank) + vec_short
        vec_ids = self.letters[:11]
        self.hhenl._bulk_label_chamber_ensemble(vecs[:10], vec_ids[:10])
        self.assertRaises(ValueError, self.hhenl._bulk_label_chamber_ensemble, *[vecs, vec_ids])

    def test_bulk_label_chamber_ensemble_2(self):
        """Throws ValueError if len(vec_ids) != len(vec_ids)."""
        self._bulk_list_length_error(self.hhenl._bulk_label_chamber_ensemble)

    def test_bulk_label_chamber_ensemble_3(self):
        """If vec_ids are all unknown, then for each hh in self.hhenl.hhs, the difference in the
        union over all chamber_ids in hh.get_chamber_ids() of hh.chamber_labels(chamber_id), before
        and after the bulk_label, is equal to vec_ids."""
        vecs = HyperplaneHasher._random_vectors(10, self.hhenl.rank)
        vec_ids = self.letters[:10]
        labels_before = [self._get_all_hh_labels(hh) for hh in self.hhenl.hhs]
        self.hhenl._bulk_label_chamber_ensemble(vecs, vec_ids)
        labels_after = [self._get_all_hh_labels(hh) for hh in self.hhenl.hhs]
        for b, a in zip(labels_before, labels_after):
            self.assertEqual(a.difference(b), set(vec_ids))

    def test_bulk_label_chamber_ensemble_4(self):
        """If vec_ids are partially known, then for each hh in self.hhenl.hhs, the difference in the
        union over all chamber_ids in hh.get_chamber_ids() of hh.chamber_labels(chamber_id), before
        and after the bulk_label, is equal to the unknown vec_ids."""
        vecs = HyperplaneHasher._random_vectors(24, self.hhenl.rank)
        old_vec_ids = self.feature_vecs_ids[:11]
        new_vec_ids = self.letters[:13]
        vec_ids = old_vec_ids + new_vec_ids
        labels_before = [self._get_all_hh_labels(hh) for hh in self.hhenl.hhs]
        self.hhenl._bulk_label_chamber_ensemble(vecs, vec_ids)
        labels_after = [self._get_all_hh_labels(hh) for hh in self.hhenl.hhs]
        for b, a in zip(labels_before, labels_after):
            self.assertEqual(a.difference(b), set(new_vec_ids))

    def test_bulk_label_chamber_ensemble_5(self):
        """Let first = [first_1, first_2, ..., first_n] and second = [second_1, second_2, ..., second_n] be
        lists of labels, and vecs = [vec_1, vec_2, ..., vec_n] a list of vectors. Then after applying the method
        first to (vecs, first), then to (vecs, second), all chambers C in all hh in self.hhenl.hhs have the property
        that first_i in C iff second_i in C."""
        vecs = HyperplaneHasher._random_vectors(20, self.hhenl.rank)
        first_ex = re.compile(r'first_([\S]*)')
        second_ex = re.compile(r'second_([\S]*)')
        first = ['first_%i' % i for i in range(20)]
        second = ['second_%i' % i for i in range(20)]
        self.hhenl._bulk_label_chamber_ensemble(vecs, first)
        self.hhenl._bulk_label_chamber_ensemble(vecs, second)
        for hh in self.hhenl.hhs:
            ch_ids = hh.get_chamber_ids()
            for ch_id in ch_ids:
                labels = hh.chamber_labels(ch_id)
                flabels = [''.join(first_ex.findall(label)) for label in labels]
                first_labels = set([entry for entry in flabels if len(entry) > 0])
                slabels = [''.join(second_ex.findall(label)) for label in labels]
                second_labels = set([entry for entry in slabels if len(entry) > 0])
                self.assertEqual(first_labels, second_labels)

    def test_get_nn_candidates_1(self):
        """Returned objects is a set of strings of length
        at least num_neighbours."""
        vec = HyperplaneHasher._random_vectors(1, self.hhenl.rank)[0]
        nn = 10
        result = self.hhenl._get_nn_candidates(vec, nn)
        self.assertIsInstance(result, set)
        for element in result:
            self.assertIsInstance(element, str)
        self.assertGreaterEqual(len(result), nn)

    def test_get_nn_candidates_2(self):
        """Throws ValueError if len(vec_ids) != len(vec_ids)."""
        self._rank_error(self.hhenl._get_nn_candidates)

    def test_get_vector_ids_1(self):
        """Fetched vector ids are the expected ones."""
        self.assertEqual(set(self.feature_vecs_ids), self.hhenl.get_vector_ids())

    def test_get_vector_1(self):
        """The returned object is a numpy array of length self.rank."""
        vec_id = self.feature_vecs_ids[0]
        vec = self.hhenl.get_vector(vec_id)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec), self.hhenl.rank)
        self.assertTrue((self.feature_vecs[0]==vec).all())

    def test_get_vector_2(self):
        """Throws KeyError if 'vec_id' is unrecognised by underlying
        KeyValueStore object."""
        vec_id = 'non_existent_vec'
        self.assertRaises(KeyError, self.hhenl.get_vector, vec_id)

    def test_bulk_get_vector_1(self):
        """The returned object is a list of numpy arrays, each of length self.rank."""
        def check_vec(vec):
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(len(vec), self.hhenl.rank)
        ids = self.feature_vecs_ids
        vecs = self.hhenl.bulk_get_vector(ids)
        self.assertIsInstance(vecs, list)
        [check_vec(vec) for vec in vecs]

    def test_bulk_get_vector_2(self):
        """Method returns a list of length equal to the number of recognised
        vector ids."""
        vec_ids = self.feature_vecs_ids
        ids_1 = vec_ids + ['non_existent_vec_%i' % i for i in range(5)]
        ids_2 = ['non_existent_vec_%i' % i for i in range(5)]
        vecs_1 = self.hhenl.bulk_get_vector(ids_1)
        vecs_2 = self.hhenl.bulk_get_vector(ids_2)
        self.assertEqual(len(vecs_1), len(vec_ids))
        self.assertEqual(len(vecs_2), 0)

    def test_bulk_get_vector_3(self):
        """Copies of the stored vectors are returned, rather than the vectors themselves.
        Thus changing any of the returned vectors does _not_ affect the stored versions."""
        vec_ids = self.feature_vecs_ids
        original = self.hhenl.bulk_get_vector(vec_ids)
        first = self.hhenl.bulk_get_vector(vec_ids)
        for vec in first:
            vec[0] = 11.0
        second = self.hhenl.bulk_get_vector(vec_ids)
        for f, s, o in zip(first, second, original):
            self.assertTrue((s == o).all())
            self.assertTrue((f != o).any())

    def test_get_rank_1(self):
        """Returns a positive integer agreeing with the length
        of a known vector, and with the underlying 'rank' attribute."""
        vec_id = self.feature_vecs_ids[0]
        vec = self.hhenl.get_vector(vec_id)
        returned_rank = self.hhenl.get_rank()
        self.assertEqual(self.hhenl.rank, returned_rank)
        self.assertEqual(len(vec), returned_rank)

    def test_delete_vector_1(self):
        """Removes 'vec' both from self.hhenl.kvstore, and from all chambers
        of all underlying HyperplaneHasher objects."""
        vec = self.feature_vecs[0]
        vec_id = self.feature_vecs_ids[0]
        self.hhenl.delete_vector(vec, vec_id)
        self.assertRaises(KeyError, self.hhenl.get_vector, vec_id)
        all_vec_ids = self.hhenl.get_vector_ids()
        self.assertNotIn(vec_id, all_vec_ids)
        for hh in self.hhenl.hhs:
            chamber_id = hh.get_chamber_id(vec)
            self.assertNotIn(vec_id, hh.chamber_labels(chamber_id))

    def test_delete_vector_2(self):
        """Throws KeyError if 'vec_id' is not a key in the underlying KeyValueStore object,
        throws ValueError if len(vec) != self.rank."""
        vec = self.feature_vecs[0]
        self._rank_error(self.hhenl.delete_vector)
        self.assertRaises(KeyError, self.hhenl.delete_vector, *[vec, 'non_existent_id'])

    def test_add_vector_1(self):
        """Adds 'vec' both to self.hhenl.kvstore, and to exactly one chamber
        of each underlying HyperplaneHasher object. Subsequently, the lists of keys of
        vectors in the objects self.hhenl.kvstore and self.hhenl.hhs[i].kvstore
        are identical, for all i."""
        vec = HyperplaneHasher._random_vectors(1, self.hhenl.rank)[0]
        vec_id = 'new'
        self.hhenl.add_vector(vec, vec_id)
        self.assertTrue((self.hhenl.get_vector(vec_id)==vec).all())
        all_vec_ids = self.hhenl.get_vector_ids()
        self.assertIn(vec_id, all_vec_ids)
        for hh in self.hhenl.hhs:
            chamber_id = hh.get_chamber_id(vec)
            self.assertIn(vec_id, hh.chamber_labels(chamber_id))

    def test_add_vector_2(self):
        """Throws ValueError if len(vec) != self.rank."""
        self._rank_error(self.hhenl.add_vector)

    def test_bulk_add_vector_1(self):
        """Throws ValueError if len(vec) != self.rank for vec in vecs."""
        self._bulk_rank_error(self.hhenl.bulk_add_vector)

    def test_bulk_add_vector_2(self):
        """Throws ValueError if len(vec) != self.rank for vec in vecs."""
        self._bulk_list_length_error(self.hhenl.bulk_add_vector)

    def _check_new_vec_ids_added(self, hhenl, vecs, vec_ids):
        """The set theoretic difference between hhenl.get_vector_ids_post and
        self.hhenl.get_vector_ids_pre is equal to the set-theoretic difference
        between set(vec_ids) and self.hhenl.get_vector_ids_pre."""
        ids_pre = self.hhenl.get_vector_ids()
        expected_diff = set(vec_ids).difference(ids_pre)
        self.hhenl.bulk_add_vector(vecs, vec_ids)
        ids_post = self.hhenl.get_vector_ids()
        actual_diff = ids_post.difference(ids_pre)
        return (actual_diff, expected_diff)

    def test_bulk_add_vector_3(self):
        """The set theoretic difference between self.hhenl.get_vector_ids_post and
        self.hhenl.get_vector_ids_pre is equal to the set of new vector ids."""
        vecs = self.feature_vecs[:10]
        vec_ids = self.letters[:10]
        new_vec_ids = self.letters[5:15]
        actual_diff, expected_diff = self._check_new_vec_ids_added(self.hhenl, vecs, vec_ids)
        self.assertEqual(actual_diff, expected_diff)
        actual_diff, expected_diff = self._check_new_vec_ids_added(self.hhenl, vecs, new_vec_ids)
        self.assertEqual(actual_diff, expected_diff)

    def test_bulk_add_vector_4(self):
        """The method is idempotent."""
        vecs = self.feature_vecs[:10]
        vec_ids = self.letters[:10]
        _, _ = self._check_new_vec_ids_added(self.hhenl, vecs, vec_ids)
        actual_diff, expected_diff = self._check_new_vec_ids_added(self.hhenl, vecs, vec_ids)
        self.assertEqual(actual_diff, set())
        self.assertEqual(actual_diff, set())

    def _chamberwise_compare(self, hhenl_1, hhenl_2):
        """Check that the chambers of all hh objects attached to each
        of hhenl_1 and hhenl_2 contain the same labels."""
        for hh_1, hh_2 in zip(hhenl_1.hhs, hhenl_2.hhs):
            hh_1_ids, hh_2_ids = hh_1.get_chamber_ids(), hh_2.get_chamber_ids()
            self.assertEqual(self._get_all_hh_labels(hh_1), self._get_all_hh_labels(hh_1))
            self.assertEqual(hh_1_ids, hh_2_ids)
            for ch_id_1, ch_id_2 in zip(hh_1_ids, hh_2_ids):
                print 'Bulk labels'
                print hh_1.chamber_labels(ch_id_1)
                print 'Serial labels'
                print hh_2.chamber_labels(ch_id_2)
                self.assertEqual(hh_1.chamber_labels(ch_id_1), hh_2.chamber_labels(ch_id_2))
                print '\n'

    def _delete_all_vectors(self, hhenl):
        """Calls delete_vector(vec_id) for every vec_id."""
        vec_ids = hhenl.get_vector_ids()
        vecs = [hhenl.get_vector(vec_id) for vec_id in vec_ids]
        for vec, vec_id in zip(vecs, vec_ids):
            hhenl.delete_vector(vec, vec_id)

    def _create_hhs_chamber_label_list(self, hhenl):
        """Returns a list [ch_label_list_1, ..., chamber_label_list_n] of lists,
        where ch_label_list_i is the list of pairs (chamber_id, labels) associated
        to the i-th HyperplaneHasher object in hhenl, and labels is the set of labels
        in chamber chamber_id."""
        hhs_ch_label_list = []
        for hh in hhenl.hhs:
            ch_ids = list(hh.get_chamber_ids())
            ch_ids.sort()
            ch_label_list = [(ch_id, hh.chamber_labels(ch_id)) for ch_id in ch_ids]
            hhs_ch_label_list.append(ch_label_list)
        return hhs_ch_label_list

    def test_bulk_add_vector_5(self):
        """Calling the method on (vecs, vec_ids) has the same effect as
        calling add_vector(vec, vec_id), for all (vec, vec_id) in
        zip(vecs, vec_ids)."""
        vecs = self.feature_vecs[:10]
        vec_ids = self.letters[:10]
        hhenl = self._create_hhenl()
        hhenl.bulk_add_vector(vecs, vec_ids)
        list_bulk = self._create_hhs_chamber_label_list(hhenl)
        self._delete_all_vectors(hhenl)
        for vec, vec_id in zip(vecs, vec_ids):
            hhenl.add_vector(vec, vec_id)
        list_serial = self._create_hhs_chamber_label_list(hhenl)
        self.assertEqual(list_bulk, list_serial)

    def test_find_neighbours_1(self):
        """Returns a pandas series of length 'num_neighbours', indexed
        by keys that can successfully be passed to the get_vector() method.
        The entries of 'ser' are non-negative real numbers, in ascending order.
        If the input vector is known to the underlying KeyValueStore object,
        then the first entry has value 0.0 and key == 'vec_id', where 'vec_id'
        is the id of the input vector."""
        vec = HyperplaneHasher._random_vectors(1, self.hhenl.rank)[0]
        nn = 10
        neighbours = self.hhenl.find_neighbours(vec, nn)
        self.assertIsInstance(neighbours, pd.Series)
        self.assertEqual(len(neighbours), nn)
        self.assertTrue((neighbours == neighbours.order()).all())
        for i in range(len(neighbours)):
            self.assertGreaterEqual(neighbours[i], 0.0)

    def test_find_neighbours_2(self):
        """If input vector 'vec' is known to underlying KeyValueStore object,
        then first entry of output has value 0.0 and key 'vec_id', the id of 'vec'."""
        vec = self.feature_vecs[0]
        vec_id = self.feature_vecs_ids[0]
        nn = 10
        neighbours = self.hhenl.find_neighbours(vec, nn)
        self.assertEqual(neighbours.iloc[0], 0.0)
        self.assertEqual(neighbours.index[0], vec_id)

    def test_find_neighbours_3(self):
        """Throws ValueError if len(vec) != self.rank."""
        self._rank_error(self.hhenl.find_neighbours)
