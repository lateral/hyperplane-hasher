import re
from key_value_store import KeyValueStore
from tools.doc_inherit import doc_inherit

INT_ID = 'int_%s'
SET_ID = 'set_%s'
VEC_ID = 'vector_%s'
INT_ID_EX = re.compile(r'int_([\S]*)')
SET_ID_EX = re.compile(r'set_([\S]*)')
VEC_ID_EX = re.compile(r'vector_([\S]*)')


class DictionaryStore(KeyValueStore):

    def __init__(self, dictionary=None):
        """Instantiates object with a dictionary.
        The empty dictionary is used if None is passed."""
        self.dictionary = dictionary if dictionary else dict()

    @doc_inherit
    def get_vector_ids(self):
        keys = self.dictionary.keys()
        return [VEC_ID_EX.findall(key)[0]
                for key in keys if len(VEC_ID_EX.findall(key)) > 0]

    @doc_inherit
    def get_int_ids(self):
        keys = self.dictionary.keys()
        return [INT_ID_EX.findall(key)[0]
                for key in keys if len(INT_ID_EX.findall(key)) > 0]

    @doc_inherit
    def get_set_ids(self):
        keys = self.dictionary.keys()
        return [SET_ID_EX.findall(key)[0]
                for key in keys if len(SET_ID_EX.findall(key)) > 0]

    @doc_inherit
    def store_vector(self, vec_id, vector):
        full_vec_id = VEC_ID % vec_id
        self.dictionary[full_vec_id] = vector.copy()

    @doc_inherit
    def get_vector(self, vec_id):
        full_vec_id = VEC_ID % vec_id
        return self.dictionary[full_vec_id].copy()

    @doc_inherit
    def bulk_get_vector(self, vec_ids):
        def vec_or_none(vec_id):
            try:
                vec = self.dictionary[vec_id].copy()
                return vec
            except KeyError:
                pass
        full_vec_ids = [VEC_ID % vec_id for vec_id in vec_ids]
        results = [vec_or_none(full_vec_id) for full_vec_id in full_vec_ids]
        return [vec for vec in results if vec is not None]

    @doc_inherit
    def remove_vector(self, vec_id):
        full_vec_id = VEC_ID % vec_id
        del self.dictionary[full_vec_id]

    @doc_inherit
    def add_to_set(self, set_id, vec_id):
        full_set_id = SET_ID % set_id
        full_vec_id = VEC_ID % vec_id
        if full_set_id in self.dictionary.keys():
            self.dictionary[full_set_id].add(full_vec_id)
        else:
            self.dictionary[full_set_id] = set([full_vec_id])

    @doc_inherit
    def remove_from_set(self, set_id, vec_id):
        full_set_id = SET_ID % set_id
        full_vec_id = VEC_ID % vec_id
        self.dictionary[full_set_id].remove(full_vec_id)

    @doc_inherit
    def remove_set(self, set_id):
        full_set_id = SET_ID % set_id
        del self.dictionary[full_set_id]

    @doc_inherit
    def get_set(self, set_id):
        full_set_id = SET_ID % set_id
        result = set([VEC_ID_EX.findall(vec_id)[0]
                      for vec_id in self.dictionary[full_set_id].copy()])
        return result

    @doc_inherit
    def get_int(self, int_id):
        full_int_id = INT_ID % int_id
        return self.dictionary[full_int_id]

    @doc_inherit
    def store_int(self, int_id, integer):
        full_int_id = INT_ID % int_id
        self.dictionary[full_int_id] = integer

    @doc_inherit
    def remove_int(self, int_id):
        full_int_id = INT_ID % int_id
        del self.dictionary[full_int_id]
