import numpy as np


class KeyValueStore(object):

    """
    All ids are strings.
    Each data type (vector, set, int, etc.) has its own namespace for ids.
    COPIES of data are stored -- for example, modifying a vector object after
    storing in a KeyValueStore has no effect on the vector stored.
    Vectors are instances of np.ndarray with dimension 1.
    The default implementations of the bulk_* methods should be overwritten in
    subclasses if a more efficient approach exists.
    """

    def get_vector_ids(self):
        """Return the set of all vector ids."""
        pass

    def get_int_ids(self):
        """Return the set of all integer ids."""
        pass

    def get_set_ids(self):
        """Return the set of all set ids."""
        pass

    def store_vector(self, vec_id, vector):
        """Stores a copy of vector 'vector' at vector id 'vec_id'.
        Overwrites if key 'vec_id' already exists."""
        pass

    def bulk_store_vector(self, vec_ids, vectors):
        """Store with (vector_id, vector) pairs given by zipping the iterables
        'vec_ids' and 'vectors'."""
        if len(vec_ids) != len(vectors):
            raise ValueError
        for vec_id, vector in zip(vec_ids, vectors):
            self.store_vector(vec_id, vector)

    def get_vector(self, vec_id):
        """Returns the vector at vector id 'vec_id'. Raises
        KeyError if key 'vec_id' is unknown."""
        pass

    def bulk_get_vector(self, vec_ids):
        """Return the vectors with vector ids given by the iterable 'vec_ids'
        as a list of numpy arrays."""
        pass

    def remove_vector(self, vec_id):
        """Removes the vector at vector id 'vec_id'. Raises
        KeyError if key 'vec_id' is unknown."""
        pass

    def add_to_set(self, set_id, element_id):
        """Adds the string 'element_id' to the set corresponding to key
        'set_id', creating this set if necessary. """
        pass

    def bulk_add_to_set(self, set_ids, element_ids):
        """For each entry (set_id, element_id) in the list 'id_value_pairs', adds
        the string 'element_id' to the set corresponding to key 'set_id', overwriting
        any existing value."""
        if len(set_ids) != len(element_ids):
            raise ValueError
        for set_id, element_id in zip(set_ids, element_ids):
            self.add_to_set(set_id, element_id)

    def remove_from_set(self, set_id, element_id):
        """Removes the string 'element_id' from the set corresponding to key
        'set_id'. KeyError is raised if 'element_id' is not in this set, or if
        no set is associated to 'set_id'."""
        pass

    def remove_set(self, set_id):
        """Removes the set corresponding to key 'set_id'. KeyError is raised if
        no set is associated to 'set_id'."""
        pass

    def get_set(self, set_id):
        """Returns a copy of the set at id 'set_id'. Raises KeyError if
        'set_id' is unknown."""
        pass

    def store_int(self, int_id, integer):
        """Stores the integer 'integer' at id 'int_id', overwriting any
        existing value."""
        pass

    def bulk_store_int(self, int_ids, integers):
        """For each entry (int_id, integer) in the list id_value_pairs, stores
        the integer 'integer' at id 'int_id', overwriting any existing value."""
        if len(int_ids) != len(integers):
            raise ValueError
        for int_id, integer in zip(int_ids, integers):
            self.store_int(int_id, integer)

    def get_int(self, int_id):
        """Returns the integer at id 'int_id'. Raises KeyError if 'int_id' is
        unknown."""
        pass

    def remove_int(self, int_id):
        """Removes the integer with id 'int_id'. Raises KeyError if 'int_id' is
        unknown."""
        pass
