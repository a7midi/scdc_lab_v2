# Claims audit (what the current data supports vs what it does not yet support)

This is a *scientific* audit meant to help you write a PRL-level narrative without over-claiming.

## What the suite does support (with current outputs)

### A. Consistency-driven *structural* organization (``Genesis``)
- Under motif-energy annealing, the directed graph rewiring **changes large-scale structure** compared to its random initial condition.
- Multiple diagnostics indicate an ``intermediate`` regime that is neither fully random nor frozen:
  - SCC condensation produces a nontrivial layered DAG.
  - Forward light-cone growth is frequently better fit by polynomial models than exponential ones (see JSON summaries).

### B. Localized, long-lived activity pockets on an emergent background (``H1``)
- In pocket-tracking runs, the system can exhibit **persistent activity localized in a small fraction of nodes**, rather than global percolation.
- The analyzer now reports fractions using `--n_total` so you can distinguish:
  - *localized pockets* (small fraction),
  - vs *percolated/filled* states (large fraction).

### C. Geodesic bias toward defect regions (lensing proxy)
- In runs with injected knots, shortest paths can pass statistically closer to the detected pocket region than in baseline runs, which is consistent with a graph-theoretic notion of geodesic focusing.

### D. Nontrivial finite symmetry images in defect-sector algebra (``H2``)
- The H2 pipeline reproducibly returns large finite groups over \(\mathbb{F}_p\) for certain (N, p, t) settings.
- Matching an order such as \(|GL(2,\mathbb{F}_{13})|=26208\) is an *interesting diagnostic* of the algebraic image, but is not by itself evidence of Standard Model gauge symmetry.

### E. Reproducible spectral ``band'' structure in specific geometries (``H3``)
- The spectral probe can yield multiple separated bands in feature space, and null models can shift/destroy the separation.

## What is *not yet* supported as a PRL-level claim

- ``We proved the Standard Model gauge group emerges.''  
  The current evidence is finite-group orders over \(\mathbb{F}_p\). PRL will require group isomorphism checks, robustness sweeps, and a credible argument connecting finite images to continuous gauge groups.

- ``We proved 3 generations of matter.''  
  A best-k=3 clustering outcome is intriguing, but PRL will require robust statistics across seeds/sizes and clear null-model controls.

- ``We simulated a Theory of Everything.''  
  This framing will almost certainly trigger desk rejection. The suite is best positioned as a **new reproducible toy model** showing how a simple consistency principle can generate geometric signatures and localized defects.

## Notes on ``excite ones`` (is it cheating?)
In a deterministic local rule system, the all-vacuum configuration is typically an absorbing fixed point. To observe propagation you must prepare a non-vacuum initial state (or add thermal noise). `--excite ones` is best described as an *initial condition* (energy injection at t=0), not a new interaction or additional law.

