#!/usr/bin/python

import unittest
import numpy as np
from nn.dynamodb_adapter import DynamoDBAdapter
from test_key_value_store import KeyValueStoreTestsAbstract
import boto.dynamodb2
from boto.dynamodb2.fields import HashKey, RangeKey
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING

class TestDynamoDBAdapter(unittest.TestCase, KeyValueStoreTestsAbstract):


    @classmethod
    def setUpClass(cls):
        conn = boto.dynamodb2.connect_to_region('eu-west-1')
        #Delete 'test_table' if it already exists.
        #While loop ensures control is returned only
        #when table is properly deleted.
        if 'test_table' in conn.list_tables()['TableNames']:
            conn.delete_table('test_table')
            while True:
                if 'test_table' not in conn.list_tables()['TableNames']:
                    break
        #Create table 'test_table'. While loop ensures thread of execution
        #regained only when table is active.
        schema = [HashKey('kind', data_type=STRING), RangeKey('id', data_type=STRING)]
        table = Table.create('test_table', connection=conn, schema=schema, throughput={'read': 5, 'write': 15})
        while True:
            if table.describe()['Table']['TableStatus'] == 'ACTIVE':
                break

    @classmethod
    def tearDownClass(cls):
        conn = boto.dynamodb2.connect_to_region('eu-west-1')
        conn.delete_table('test_table')
        while True:
            if 'test_table' not in conn.list_tables()['TableNames']:
                break

    def setUp(self):
        self.store = DynamoDBAdapter(np.dtype('float32'), 'test_table')
        KeyValueStoreTestsAbstract.setUp(self)

    def tearDown(self):
        [item.delete() for item in self.store.table.scan()]
