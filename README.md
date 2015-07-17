# hyperplane-hasher
Implementation of an algorithm computing the nearest "N" neighbours to a vector, using a collection of hyperplane hashers. The vision we have for this tool is the following:

(i) It should be fast and accurate.
(ii) It should be stable and tested in a production environment.
(iii) You should be able to control the number of recommendations it returns.
(iv) It should be extensible in real time: You don’t have to shut down the server it runs on and re-index every time you add a new vector to the corpus.
(v) It should be scalable: it works quickly no matter how many vectors you have to search through.
(vi) It should operate out-of-memory: The size of the set of vectors from which you can recommend isn’t limited by how much memory your machine has.

How are we doing so far? At the moment, it isn’t as fast as we’d like, we’ve not benchmarked its accuracy and it’s not production-tested. However, it does satisfy all other requirements: users can determine the number of recommendations on a per-call basis, vectors can be added with no downtime, it’s scalable and it operates out-of-memory using a DynamoDB adaptor.

How does it work? There is a hierarchy comprising three classes. 

(i) On the bottom layer is the KeyValueStore class. It is a shell superclass, designed to store, delete and retrieve three kinds of objects: vectors, sets and integers. It has two concrete subclasses at present: (a) DictionaryStore and (b)DynamoDBAdaptor; these store underlying values in a Python dictionary and a DynamoDB table, respectively. 

(ii) In the middle layer is the HyperplaneHasher class. It takes a KeyValueStore object kvstore as an initialisation parameter, and contains methods to compute the chamber to which a vector belongs, as well as include remove and retrieve “labels” — strings — in particular chambers. The collection of labels contained in a chamber is stored as a set in the underlying kvstore. In practice, one often computes the id of a chamber to which a vector belongs, and then labels that chamber with the vector id. 

(iii) The top layers is the HHEnsembleNNLookup class. It takes a KeyValueStore object kvstore as an initialisation parameter, as well as the number of hashers to be created and the number of hyperplanes in each. It contains methods to add, retrieve and remove vectors. Adding a vector writes both the vector itself to the underlying KeyValueStore object, as well as writing its id to the appropriate chamber for each hyperplane hasher. The find_neighbours() method finds NN candidates according to each HyperplaneHasher instance, throws them into a big pool, then returns the ids of the nearest “N” to the given input vector. 

What can be fixed?

(i) The hashers are queried on multiple threads using the python "threading" module. With 10 hashers on about 100,000 vectors of dimension 400, this results in a factor-of-two speedup (from around 1 second to 500ms) when compared to sequential querying with no threads. However, if it were properly asynchronous, I'd expect the speedup to be closer to the number of hashers -- around 10x. Perhaps this can be achieved with a properly asynchronous framework like Tornado?

(ii) At present, the hyperplanes are randomly created when the HHEnsembleNNLookup object is created, and are forgotten when the machine is turned off. This means that all vectors must be re-hashed every time a HHEnsembleNNLookup object is initialised from a given set of vectors in DynamoDB. It would be better if the normal vectors could also be stored, preventing the need to re-hash in this way.
