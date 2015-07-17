import unittest, string, random as rd, numpy as np
from nn.dictionary_store import DictionaryStore
from ann.hyperplane_hasher import *
import boto.dynamodb2
from nn.dynamodb_adapter import DynamoDBAdapter
from boto.dynamodb2.fields import HashKey, RangeKey
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING

VISIBLE_CHAMBERS = 20
NUM_NORMAL_VECS = 5
DIM  = 10
NAME = 'test_HH'

class HyperplaneHasherTestAbstract(object):

    def setUp(self):
        self.valid_test_vector = self._random_vectors(1, DIM)[0]
        self.invalid_test_vector = self._random_vectors(1, DIM - 1)[0]
        self.hh = self._create_hh()
        self.letters = list(string.ascii_lowercase)
        self._fill_chambers()

    def _create_hh(self):
        normal_vectors = self._random_vectors(NUM_NORMAL_VECS, DIM)
        return HyperplaneHasher(normal_vectors=normal_vectors, kvstore=self.kvstore, name=NAME)

    def _fill_chambers(self):
        """Fills all except one chamber with labels."""
        self.valid_chamber_ids = self.hh._all_binary_strings()[:VISIBLE_CHAMBERS]
        cid = self.hh.get_chamber_id(self.valid_test_vector)
        if cid not in self.valid_chamber_ids:
            self.valid_chamber_ids = [cid] + self.valid_chamber_ids[1:]
        self.valid_chamber_id = self.valid_chamber_ids[0]
        self.empty_chamber_id = self.valid_chamber_ids[-1]
        num_chambers = len(self.valid_chamber_ids)
        self.total_labels_num = num_chambers * 10
        for i in range(self.total_labels_num):
            self.hh.label_chamber(self.valid_chamber_ids[i % (num_chambers - 1)], 'label_%i' % i)
            self.num_chambers = num_chambers

    def _random_vectors(self, num, dim):
        """Generates 'num' random vectors of rank 'rank'."""
        return [np.array([rd.random() - 0.5 for i in range(dim)]) for j in range(num)]

    def _hyperplane_reflection(self, vector, normal_vector):
        """Computes the reflection of 'vector' about the
        hyperplane defined by 'normal_vector'."""
        return vector - 2 * (np.dot(vector, normal_vector)/np.dot(normal_vector, normal_vector)) * normal_vector

    def test_init_1(self):
        """Two identically initialised HyperplaneHasher objects
        assign the same chamber ids to the same vectors."""
        normal_vectors = self._random_vectors(NUM_NORMAL_VECS, DIM)
        kvstore = DictionaryStore(dict())
        hh_1 = HyperplaneHasher(normal_vectors=normal_vectors, kvstore=kvstore, name=NAME)
        hh_2 = HyperplaneHasher(normal_vectors=normal_vectors, kvstore=kvstore, name=NAME)
        vecs = self._random_vectors(10, DIM)
        for vec in vecs:
            self.assertEquals(hh_1.get_chamber_id(vec), hh_2.get_chamber_id(vec))

    def test_label_chamber_1(self):
        """If chamber_id unknown, then creates a new chamber and labels it."""
        self.hh.label_chamber(self.empty_chamber_id, 'label')
        self.assertIn('label', self.hh.chamber_labels(self.empty_chamber_id))

    def test_label_chamber_2(self):
        """If chamber_id is valid, then label in chamber_labels(chamber_id)."""
        self.hh.label_chamber(self.valid_chamber_id, 'label')
        self.assertIn('label', self.hh.chamber_labels(self.valid_chamber_id))

    def test_label_chamber_3(self):
        """Adding a label to a chamber twice changes nothing."""
        self.hh.label_chamber(self.valid_chamber_id, 'label')
        first = self.hh.chamber_labels(self.valid_chamber_id)
        self.hh.label_chamber(self.valid_chamber_id, 'label')
        second = self.hh.chamber_labels(self.valid_chamber_id)
        self.assertEquals(first, second)

    def test_bulk_label_chamber_1(self):
        """Raises ValueError if len(chamber_ids) != len(labels)."""
        chamber_ids, labels = ['chamber_1', 'chamber_2', 'chamber_3'], ['label_1', 'label_2']
        self.assertRaises(ValueError, self.hh.bulk_label_chamber, *[chamber_ids, labels])

    def test_bulk_label_chamber_2(self):
        """bulk_label_chamber(chamber_ids, label_ids) has same effect as
        label_chamber(chamber_id, label_id) for all (chamber_id, label_id)
        in chamber_ids x label_ids."""
        chamber_ids = [self.letters[i] for i in range(10)]*10
        labels = ['label_%i' % i for i in range(100)]
        hh_bulk, hh_serial = self._create_hh(), self._create_hh()
        hh_bulk.bulk_label_chamber(chamber_ids, labels)
        hh_serial.bulk_label_chamber(chamber_ids, labels)
        bulk_chamber_ids, serial_chamber_ids = hh_bulk.get_chamber_ids(), hh_serial.get_chamber_ids()
        self.assertEqual(bulk_chamber_ids, serial_chamber_ids)
        for chamber_id in bulk_chamber_ids:
            self.assertEqual(hh_bulk.chamber_labels(chamber_id),
                            hh_serial.chamber_labels(chamber_id))

    def _make_chamber_ids_labels(self, span):
        chamber_ids = [self.letters[i] for i in span]
        labels = ['label_%i' % i for i in span]
        return (chamber_ids, labels)

    def _actual_expected_id_diff(self, hh, chamber_ids, labels):
        expected_diff = set(chamber_ids).difference(hh.get_chamber_ids())
        old_ids = hh.get_chamber_ids()
        hh.bulk_label_chamber(chamber_ids, labels)
        new_ids = hh.get_chamber_ids()
        actual_diff = new_ids.difference(old_ids)
        return (expected_diff, actual_diff)

    def test_bulk_label_chamber_3(self):
        """Set theoretic difference between the pre- and post-values of
        get_chamber_ids() is equal to the set of all unseen input chamber_ids."""
        hh = self._create_hh()
        chamber_ids, labels = self._make_chamber_ids_labels(range(6))
        expected_diff, actual_diff = self._actual_expected_id_diff(hh, chamber_ids, labels)
        self.assertEqual(actual_diff, expected_diff)
        chamber_ids, labels = self._make_chamber_ids_labels(range(4, 8))
        expected_diff, actual_diff = self._actual_expected_id_diff(hh, chamber_ids, labels)
        self.assertEqual(actual_diff, expected_diff)

    def test_bulk_label_chamber_4(self):
        """Let X = zip(chamber_ids, labels) and let Y = X with repeat tuples removed.i
        Then bulk_label_chamber(unzip(X)) = bulk_label_chamber(unzip(Y))."""
        pass

    def test_unlabel_chamber_1(self):
        """Throws KeyError if label is not an element of
        the chamber associated to chamber_id."""
        self.assertRaises(KeyError, self.hh.unlabel_chamber, *[self.valid_chamber_id, 'non_existent_label'])

    def test_unlabel_chamber_2(self):
        """Removes label from chamber if chamber_id exists
        and label belongs to chamber."""
        self.hh.label_chamber(self.valid_chamber_id, 'label')
        first = self.hh.chamber_labels(self.valid_chamber_id).copy()
        self.hh.unlabel_chamber(self.valid_chamber_id, 'label')
        second = self.hh.chamber_labels(self.valid_chamber_id).copy()
        self.assertEquals(first.difference(second), set(['label']))

    def test_chamber_labels_1(self):
        """Returns a set of strings."""
        self.hh.label_chamber(self.valid_chamber_id, 'label')
        cl = self.hh.chamber_labels(self.valid_chamber_id)
        self.assertIsInstance(cl, set)
        for label in cl:
            self.assertIsInstance(label, str)

    def test_chamber_labels_2(self):
        """Returns empty set when 'chamber_id' is unknown."""
        self.assertEquals(set(), self.hh.chamber_labels(self.empty_chamber_id))

    def test_get_chamber_id_1(self):
        """Throws ValueError exception if rank(vector) != rank of normal vectors."""
        self.assertRaises(ValueError, self.hh.get_chamber_id, self.invalid_test_vector)

    def test_get_chamber_id_2(self):
        """Returned chamber_id is a string of length == num(normal_vectors),
        consisting of 1s and 0s."""
        cid = self.hh.get_chamber_id(self.valid_test_vector)
        self.assertIsInstance(cid, str)
        self.assertEquals(len(cid), self.hh.num_normal_vectors)
        for letter in cid:
            self.assertIn(letter, ['0', '1'])

    def test_get_chamber_id_3(self):
        """Returned chamber_id belongs to list of valid chamber ids."""
        cid = self.hh.get_chamber_id(self.valid_test_vector)
        self.assertIn(cid, self.valid_chamber_ids)

    def test_get_chamber_id_4(self):
        """Output changes if input vector changed so that its dot
        product with a normal vector changes."""
        chamber_id_1 = self.hh.get_chamber_id(self.valid_test_vector)
        reflected_vector = self._hyperplane_reflection(self.valid_test_vector, self.hh.get_normal_vectors()[0])
        chamber_id_2 = self.hh.get_chamber_id(reflected_vector)
        self.assertNotEqual(chamber_id_1, chamber_id_2)

    def test_get_chamber_id_5(self):
        """The output is correct on a precomputed HH object
        in 2 dimensions and with 6 chambers, given as input
        a collection of vectors lying in all chambers."""
        RTT = np.sqrt(3.0)/2.0
        a = np.array([0, 1])
        b = np.array([-RTT, 0.5])
        c = np.array([-RTT, -0.5])
        normal_vectors = [a, b, c]
        kvstore = DictionaryStore(dict())
        hh = HyperplaneHasher(normal_vectors=normal_vectors, kvstore=kvstore, name='hexagon')
        vecs = [-c, a, b, c, -a, -b]
        answers = ['100', '110', '111', '011', '001', '000']
        for i, vec in enumerate(vecs):
            self.assertEquals(hh.get_chamber_id(vec), answers[i])

    def test_adjacent_chamber_ids_1(self):
        """Works as expected, even when chamber is empty."""
        self.hh.adjacent_chamber_ids(self.empty_chamber_id)

    def test_adjacent_chamber_ids_2(self):
        """Returned list contains chamber_id if chamber_id valid."""
        ncids = self.hh.adjacent_chamber_ids(self.valid_chamber_id)
        self.assertIn(self.valid_chamber_id, ncids)

    def test_proximal_chamber_ids_1(self):
        """Works as expected, even when chamber is empty."""
        self.hh.proximal_chamber_ids(self.empty_chamber_id, 5)

    def test_proximal_chamber_ids_3(self):
        """No proper subchain of the outputted ids corresponds
        to a collection having more than 'num_labels' labels.
        However, the whole collection has at least 'num_labels' labels."""
        num_labels = self.total_labels_num/2
        pcids = self.hh.proximal_chamber_ids(self.valid_chamber_id, num_labels)
        counts = [(pcid, len(self.hh.chamber_labels(pcid))) for pcid in pcids]
        total = 0
        for i, entry in enumerate(counts):
            total += counts[i][1]
            if i != len(counts) - 1:
                self.assertLess(total, num_labels)
            else:
                self.assertGreaterEqual(total, num_labels)

    def test_proximal_chamber_ids_4(self):
        """chamber_id is an element of the output."""
        num_labels = self.total_labels_num/2
        pcids = self.hh.proximal_chamber_ids(self.valid_chamber_id, num_labels)
        self.assertIn(self.valid_chamber_id, pcids)

    def test_proximal_chamber_ids_5(self):
        """All chambers in outputted list have non-identical direct
        neighbours also in the list."""
        num_labels = self.total_labels_num/2
        pcids = self.hh.proximal_chamber_ids(self.valid_chamber_id, num_labels)
        for pcid in pcids:
            neighbours = self.hh.adjacent_chamber_ids(pcid)
            proximal_neighbours = (set(neighbours).intersection(set(pcids))).difference(set([pcid]))
            self.assertNotEqual(proximal_neighbours, set())

    def test_proximal_chamber_labels_1(self):
        """Works as normal, even if chamber is empty."""
        self.hh.proximal_chamber_labels(self.empty_chamber_id, 5)


    def test_proximal_chamber_labels_3(self):
        """chamber_labels(self, chamber_id) is a subset of
        proximal_chamber_labels(self, chamber_id, num_labels)."""
        cls = self.hh.chamber_labels(self.valid_chamber_id)
        pcls = self.hh.proximal_chamber_labels(self.valid_chamber_id, 10)
        self.assertTrue(cls.issubset(pcls))

    def test_proximal_chamber_labels_4(self):
        """Every outputted label belongs to at least one chamber
        in the output of proximal_chamber_ids(chamber_id)."""
        pcls = self.hh.proximal_chamber_labels(self.valid_chamber_id, 30)
        pcids = self.hh.proximal_chamber_ids(self.valid_chamber_id, 30)
        for label in pcls:
            bools = [(label in self.hh.chamber_labels(pcid)) for pcid in pcids]
            self.assertTrue(reduce(lambda x, y: x or y, bools))

    def test_get_normal_vectors_1(self):
        """Returns a list of numpy arrays, all 1-dimensional
        and all of same rank."""
        nvecs = self.hh.get_normal_vectors()
        vector_rank = len(nvecs[0])
        self.assertIsInstance(nvecs, list)
        for vec in nvecs:
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(len(vec.shape), 1)
            self.assertEqual(len(vec), vector_rank)

    def test_random_vectors_1(self):
        """Returns a length num list of numpy arrays, all
        1-dimensional and all of same rank."""
        num = 10
        rank = 20
        rvecs = self.hh._random_vectors(num, rank)
        self.assertIsInstance(rvecs, list)
        for vec in rvecs:
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(len(vec.shape), 1)
            self.assertEqual(len(vec), rank)

    def test_compute_num_chambers(self):
        """Output agrees with precomputed output for various
        values of d = dimension and n = number of hyperplanes."""
        #pairs of the form (d, n)
        pairs = [(2, 1), (2, 2), (3, 3), (3, 4), (4, 4)]
        answers = [2, 4, 8, 14, 16]
        for i, pair in enumerate(pairs):
            d, n = pair
            normal_vectors = self._random_vectors(n, d)
            kvstore = DictionaryStore(dict())
            hh = HyperplaneHasher(normal_vectors=normal_vectors, kvstore=kvstore, name=NAME)
            self.assertEquals(hh._compute_num_chambers(), answers[i])

    def test_flip_digit_1(self):
        """Throws ValueError if first argument does not consist
        soley of 1s and 0s. Throws a ValueError if the second argument
        is a negative integer, or an integer exceeding
        len(first_argument) - 1."""
        self.assertRaises(ValueError, self.hh._flip_digit, *['ababa', 1])
        self.assertRaises(ValueError, self.hh._flip_digit, *['101010', -1])
        self.assertRaises(ValueError, self.hh._flip_digit, *['101010', 10])

    def test_flip_digit_2(self):
        """Yields correct answers if input is a string."""
        self.assertEquals(self.hh._flip_digit('101010', 1), '111010')
        self.assertEquals(self.hh._flip_digit('101010', 0), '001010')
        self.assertEquals(self.hh._flip_digit('101010', 2), '100010')
        self.assertEquals(self.hh._flip_digit('101010', 3), '101110')

    def test_hamming_distance_1(self):
        """Throws ValueError if input arguments are not strings consisting
        soley of 1s and 0s. Throws a ValueError if one string is longer
        than the other."""
        self.assertRaises(ValueError, self.hh._hamming_distance, *['abab', '1010'])
        self.assertRaises(ValueError, self.hh._hamming_distance, *['1010', 'cdcd'])
        self.assertRaises(ValueError, self.hh._hamming_distance, *['1010', '10101'])

    def test_hamming_distance_2(self):
        """Returns the correct answer given valid input."""
        self.assertEquals(self.hh._hamming_distance('10101', '10101'), 0)
        self.assertEquals(self.hh._hamming_distance('10001', '10101'), 1)
        self.assertEquals(self.hh._hamming_distance('11111', '10101'), 2)
        self.assertEquals(self.hh._hamming_distance('01010', '10101'), 5)

    def test_hamming_distance_i_1(self):
        """Throws a ValueError if first argument is not a string of correct length,
        or does not consist soley of 1s and 0s. Throws a ValueError if second argument
        is either negative or exceeds the number of normal vectors."""
        self.assertRaises(ValueError, self.hh._hamming_distance_i, *['010110', 2])
        self.assertRaises(ValueError, self.hh._hamming_distance_i, *['ab010', 2])
        self.assertRaises(ValueError, self.hh._hamming_distance_i, *['11010', 10])
        self.assertRaises(ValueError, self.hh._hamming_distance_i, *['11010', -3])

    def test_hamming_distance_i_2(self):
        """Returns a sorted list of strings of correct length, consisting soley
        of 1s and 0s. The Hamming Distance of all strings from the input equals i."""
        input_string = '11010'
        i = 3
        results = self.hh._hamming_distance_i(input_string, 3)
        self.assertEquals(results, sorted(results))
        for result in results:
            self.assertEquals(len(result), self.hh.num_normal_vectors)
            self.assertEquals(self.hh._hamming_distance(input_string, result), i)
            for letter in result:
                self.assertIn(letter, ['0', '1'])

    def test_all_binary_strings(self):
        """The output is an alphabetically sorted list of strings consisting
        of 2**n entries, where n == self.hh.num_normal_vectors. Each entry is
        a string of length n consisting soley of 0s and 1s."""
        n = self.hh.num_normal_vectors
        bs = self.hh._all_binary_strings()
        self.assertEquals(bs, sorted(bs))
        self.assertIsInstance(bs, list)
        self.assertEquals(len(bs), 2**n)
        for string in bs:
            self.assertIsInstance(string, str)
            self.assertEquals(len(string), n)
            for letter in string:
                self.assertIn(letter, ['0', '1'])

    def test_random_vectors(self):
        """Returns a list of numpy arrays of length 'num', each
        of dimension 'dim'."""
        num = 5
        dim = 10
        results = self.hh._random_vectors(num, dim)
        self.assertIsInstance(results, list)
        self.assertEquals(len(results), num)
        for vec in results:
            self.assertIsInstance(vec, np.ndarray)
            self.assertEquals(len(vec), dim)

    def test_get_chamber_ids_1(self):
        """Returns a set of strings."""
        chamber_ids = self.hh.get_chamber_ids()
        self.assertIsInstance(chamber_ids, set)
        for chamber_id in chamber_ids:
            self.assertIsInstance(chamber_id, str)

class TestHyperplaneHasherDictionary(unittest.TestCase, HyperplaneHasherTestAbstract):

    def setUp(self):
        self.kvstore = DictionaryStore(dict())
        HyperplaneHasherTestAbstract.setUp(self)
