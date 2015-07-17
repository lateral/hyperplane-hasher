import boto.dynamodb
from boto.dynamodb.types import Binary
from boto.dynamodb2.exceptions import ItemNotFound
from boto.dynamodb2.fields import HashKey, KeysOnlyIndex, RangeKey
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.table import Item, Table
from boto.dynamodb2.types import NUMBER, STRING
from boto.exception import JSONResponseError
import numpy as np
import nn.key_value_store as key_value_store


class DynamoDBAdapter(key_value_store.KeyValueStore):

    """ Implementation of an abstract key-value store defined in
    key_value_store.py. The underlying database is amazon DynamoDB.

    The store keeps all objects in a single table with following schema:
    [HashKey('kind', data_type=STRING), RangeKey('id')]. 'kind' is the string
    with the object type ('vector', 'set' or 'int') and 'id' is the object id.
    The object value is stored in the 'value' attribute of the table items.

    The table should be created before this code is executed. Amazon
    configuration is assumed to be stored in ~/.boto file as described in
    http://boto.readthedocs.org/en/latest/boto_config_tut.html
    """

    def __init__(self, precision=np.dtype('float32'), table_name='test'):
        """ Create an instance of the dynamodb key-value store.
        precision - a numpy type, elements of all vectors are converted and
           stored in this type;
        table_name - the name of the DynamoDB table which keeps the objects.
        """
        conn = boto.dynamodb2.connect_to_region('eu-west-1')
        if not isinstance(precision, np.dtype):
            raise TypeError("Precision should be a numpy.dtype subtype")
        self.precision = precision
        self.precision_name = precision.name
        self.table = Table(table_name, connection=conn)

    def _get_or_create_item(self, kind, item_id):
        try:
            item = self.table.get_item(kind=kind, id=item_id)
        except ItemNotFound:
            item = Item(self.table)
            item['kind'] = kind
            item['id'] = item_id
        return item

    def _create_vector_item(self, vec_id, vector):
        item = self._get_or_create_item('vector', vec_id)
        item['value'] = Binary(vector.astype(self.precision).tostring())
        item['precision'] = self.precision_name
        return item

    def _vector_value(self, item):
        return np.fromstring(str(item['value']), np.dtype(item['precision']))

    def get_vector_ids(self):
        return [v['id'] for v in self.table.query_2(kind__eq='vector')]

    def get_int_ids(self):
        return [v['id'] for v in self.table.query_2(kind__eq='int')]

    def get_set_ids(self):
        return [v['id'] for v in self.table.query_2(kind__eq='set')]

    def store_vector(self, vec_id, vector):
        item = self._create_vector_item(vec_id, vector)
        item.save()

    def get_vector(self, vec_id):
        try:
            item = self.table.get_item(kind='vector', id=vec_id)
        except ItemNotFound:
            raise KeyError('Vector key %s is unknown' % (vec_id,))
        return self._vector_value(item)

    def bulk_get_vector(self, vec_ids):
        keys = [{'kind': 'vector', 'id': i} for i in vec_ids]
        vs = self.table.batch_get(keys=keys)
        return [self._vector_value(i) for i in vs]

    def remove_vector(self, vec_id):
        try:
            item = self.table.get_item(kind='vector', id=vec_id)
        except ItemNotFound:
            raise KeyError('Vector key %s is unknown' % (vec_id,))
        item.delete()

    def add_to_set(self, set_id, element_id):
        item = self._get_or_create_item('set', set_id)
        if 'value' not in item.keys() or not isinstance(item['value'], set):
            item['value'] = set()
        item['value'].add(element_id)
        item.save(overwrite=True)

    def remove_from_set(self, set_id, element_id):
        try:
            item = self.table.get_item(kind='set', id=set_id)
        except ItemNotFound:
            raise KeyError('Set key %s is unknown' % (set_id,))
        if 'value' not in item.keys() or not isinstance(item['value'], set):
            raise KeyError('Incorrect value in item %s' % (set_id,))
        if element_id not in item['value']:
            raise KeyError('Element %s not in set %s' % (element_id, set_id))
        item['value'].remove(element_id)
        item.save()

    def remove_set(self, set_id):
        try:
            item = self.table.get_item(kind='set', id=set_id)
            item.delete()
        except ItemNotFound:
            raise KeyError('Set key %s is unknown' % (set_id,))

    def get_set(self, set_id):
        try:
            the_set = self.table.get_item(kind='set', id=set_id)['value']
            return set([str(entry) for entry in the_set])
        except ItemNotFound:
            raise KeyError('Set key %s is unknown' % (set_id,))

    def store_int(self, int_id, integer):
        item = self._get_or_create_item('int', int_id)
        item['value'] = integer
        item.save()

    def get_int(self, int_id):
        try:
            return int(self.table.get_item(kind='int', id=int_id)['value'])
        except ItemNotFound:
            raise KeyError('Int key %s is unknown' % (int_id,))

    def remove_int(self, int_id):
        try:
            item = self.table.get_item(kind='int', id=int_id)
        except ItemNotFound:
            raise KeyError('Int key %s is unknown' % (int_id,))
        item.delete()

    def _aggregate_set_id_element_pairs(self, setpairs):
        """Turns a list of pairs of the form (set_id, element_id) into a list 'L' of
        pairs 'p' of the form (set_id, set_of_element_ids). 'L' has the property
        that if 'p' and 'q' are distinct entries in 'L', then p[0] and q[0] are
        also distinct."""
        set_ids = set([entry[0] for entry in setpairs])
        listlist = [[entry for entry in setpairs if entry[0] == set_id]
                    for set_id in set_ids]
        result = [(pairlist[0][0], set([entry[1] for entry in pairlist]))
                  for pairlist in listlist]
        return result

    def bulk_store_vector(self, vec_ids, vectors):
        if len(vec_ids) != len(vectors):
            raise ValueError
        vecpairs = zip(vec_ids, vectors)
        with self.table.batch_write() as batch:
            for vec_id, vec in vecpairs:
                item = self._create_vector_item(vec_id, vec)
                batch.put_item(item)

    def bulk_store_vector_old(self, vectors_df):
        """Argument 'vectors' is a dataframe with index vector ids."""
        if len(vec_ids) != len(vectors):
            raise ValueError
        with self.table.batch_write() as batch:
            for ind in vectors_df.index:
                vec_id = str(ind)
                vec = vectors_df.loc[ind].values
                item = self._create_vector_item(vec_id, vec)
                batch.put_item(item)

    def bulk_store_int(self, int_ids, integers):
        """Argument 'intpairs' is a list of pairs of the form (int_id, integer)."""
        if len(int_ids) != len(integers):
            raise ValueError
        intpairs = zip(int_ids, integers)
        with self.table.batch_write() as batch:
            for pair in intpairs:
                int_id, integer = pair
                item = self._get_or_create_item('int', int_id)
                item['value'] = integer
                batch.put_item(item)

    def bulk_add_to_set(self, set_ids, element_ids):
        """batch_write() objects if the same item is written to more
        than once per batch, hence we aggregate all (set_id, element_id)
        pairs into a list of pairs (set_id, element_ids), where
        the 'set_id's are pairwise distinct, and the 'element_ids'
        are sets."""
        if len(set_ids) != len(element_ids):
            raise ValueError
        setpairs = zip(set_ids, element_ids)
        setlist = self._aggregate_set_id_element_pairs(setpairs)
        with self.table.batch_write() as batch:
            for pair in setlist:
                set_id, element_ids = pair
                item = self._get_or_create_item('set', set_id)
                if 'value' not in item.keys() or not isinstance(
                        item['value'], set):
                    item['value'] = set()
                item['value'].update(element_ids)
                batch.put_item(item)
