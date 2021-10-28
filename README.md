#### Graph Representation Learning

---

#### TODO
- [ ] handy embedding evaluation method (with eigen as a reference): encode-decode-evaluate
- [ ] refactor redundant embed API 
- [ ] AUC and accuracy computed in numba-jitted functions.

#### BUGS
- [ ] shallow embedding causes segmentation fault on EC2 machines, but not on Hetzner machines

#### DOCUMENTATION 
- [ ] Sphinx docstrings & RTD stub. 
- [ ] List reasons to use 1-indexing.
- [ ] grl vs tf benchmarks

#### OTHER
- [ ] is what's currently in `jobs` really belong to this repository? 
- [ ] port graph to shmem 
- [ ] Global link prediction heuristics. 
- [ ] Built-in real-world datasets.  

---

Other stuff;
* Functions defined in `graph.core` are exported at the module level. 
* I try to mark places where conversions between 1- and 0- indexing are taking place with a `@indexing` comment. 
* Convention: functions that modify their arguments in place (without a memcpy) do not return anything. 
* Many tests are carried out against `igraph`, `numpy` and `tensorflow`. This is not very elegant, and shouldn't be *the only* case, but for initial development should suffice.  
