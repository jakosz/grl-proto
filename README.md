#### Graph Representation Learning

---

#### TODO
- [x] Conversions between three formats: igraph, adjacency, native.
- [ ] Sphinx docstrings & RTD stub. 
- [ ] List reasons to use 1-indexing.
- [ ] AUC and accuracy computed in numba-jitted functions.
- [ ] is what's currently in `jobs` really belong to this repository? 
- [ ] port graph to shmem 

0.5
- [ ] all three flavours of shallow embedding implemented in grl 
- [ ] Conversions to/from OGB.

- [ ] Global link prediction heuristics. 
- [ ] Built-in real-world datasets.  

---

Other stuff;
* Functions defined in `graph.core` are exported at the module level. 
* I try to mark places where conversions between 1- and 0- indexing are taking place with a `@indexing` comment. 
* Convention: functions that modify their arguments in place (without a memcpy) do not return anything. 
* Many tests are carried out against `igraph`, `numpy` and `tensorflow`. This is not very elegant, and shouldn't be *the only* case, but for initial development should suffice.  
