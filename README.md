#### Graph Representation Learning

---

#### TODO
- [ ] diagonal model is 2x slower than symmetric and asymmetric. it shouldn't be. investigate.  
- [ ] evaluation for larger graphs should not rely on full adjacency matrix. 
- [ ] wrap encode-decode-evaluate in one function. do I want to wrap embeddings in an object?
- [ ] AUC and accuracy computed in numba-jitted functions.
- [ ] port graph to shmem.

#### DOCUMENTATION 
- [ ] Sphinx, docstrings & RTD. 
- [ ] List reasons to use 1-indexing.
- [ ] grl vs tf benchmarks.
- [ ] learning rate & number of steps conditional on graph size/density.

#### OTHER
- [ ] Global link prediction heuristics. 
- [ ] Built-in real-world datasets.  

---

Other stuff;
* Functions defined in `graph.core` are exported at the module level. 
* I try to mark places where conversions between 1- and 0- indexing are taking place with a `@indexing` comment. 
* Convention: functions that modify their arguments in place (without a memcpy) do not return anything. 
* Many tests are carried out against `igraph`, `numpy` and `tensorflow`. This is not very elegant, and shouldn't be *the only* case, but for initial development should suffice.  
