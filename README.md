# Horizontal Diffusion Maps for Redistricting Ensembles

Jenny Chan. Duke Math+ 2026, *Localized Gerrymandering* project, with Greg Herschlag and Biji Wong.

## Abstract

Ensemble methods can show that a whole districting plan is a partisan outlier, but they say much less about which district is responsible. Answering that takes a notion of which districts are similar, and a district is more than one point in a dataset: it is a set of precincts, and which precincts two districts share matters. We adapt Horizontal Diffusion Maps (Gao, 2019), a framework developed for collections of objects with internal structure such as primate teeth, to precinct-district data. This repository contains my sparse Python implementation of the horizontal operator, checked on a 97-plan Connecticut ensemble of 67,415 precinct-district pairs. No lemurs were redistricted in the making of this repo.

## Motivation

The standard test for gerrymandering generates a large ensemble of neutral maps and asks where the enacted map falls among them. That works at the level of a whole plan. A local version asks how districts like this one usually vote, which first requires knowing what "like this one" means.

Classical diffusion maps can learn that similarity, but they treat each district as a single point and keep only a distance between pairs. HDM keeps the correspondence behind that distance. Gao builds it from continuous maps between tooth surfaces. Districts have no such map, but they do have a discrete analogue, which is the precincts they share.

## Methodology

A point in the data is a pair (p, D) with precinct p in district D, and the walk moves in two ways:

- **Horizontal:** (p, A) → (p, B). Same precinct, different district. The shared precinct plays the role of Gao's parallel transport.
- **Vertical:** (p, D) → (q, D). Neighboring precincts in the same district.

The full transition matrix combines the two, with ω weighting vertical movement:

$$T = \omega D_V^{-1} A_V + (1-\omega) D_W^{-1} W$$

This repository builds the horizontal term in five steps. The matrices are sparse throughout, since almost no pairs of points share an edge.

1. **Graph.** For each precinct, link each district containing it to its k nearest neighbors under Jaccard distance, $d(A,B) = |A \triangle B| / |A \cup B|$.
2. **Kernel.** Weight each edge by $W_{ij} = \exp(-b \, d(i,j)^2)$ and symmetrize.
3. **Density normalization.** Set $W_\alpha = D^{-\alpha} W D^{-\alpha}$. Here α = 0 keeps the sampling density of the ensemble and α = 1 removes it (Coifman and Lafon, 2006).
4. **Markov operator.** Row-normalize to get $P = D_\alpha^{-1} W_\alpha$.
5. **Spectrum.** Diagonalize the symmetric conjugate $D_\alpha^{-1/2} W_\alpha D_\alpha^{-1/2}$ and map its eigenvectors back to those of P.

## Results so far

On the Connecticut ensemble (97 plans, 6 districts each, 695 precincts), the operator builds in under 20 seconds and P has a density of 2 × 10⁻⁴.

Horizontal steps alone never leave a precinct, so P is reducible, with 783 connected components and an eigenvalue of 1 for each. The walker is loyal to its precinct to a fault. This means `embed()` on the horizontal term only labels components, mapping all 67,415 points onto 783 locations. That is the expected behavior, and it is the reason the vertical term matters: the embedding becomes informative only once T connects the components.

## Next steps

- Build the vertical operator and tune ω.
- Compare against the project's Metropolis-Hastings construction, which fixes the stationary distribution of the walk using a diffusion map on districts and population-weighted district cores from a Poisson equation. That work is in the team's final report and is not implemented here.
- Replace Jaccard distance with one that uses vote data, so that closeness reflects political similarity as well as geographic overlap.

## Usage

```bash
pip install -r requirements.txt
pytest
```

```python
from horizontal_diffusion_map import HorizontalDiffusionMap

# data[j] has shape (num_districts_containing_j, num_precincts):
# each row is a 0/1 encoding of one district that contains precinct j
hdm = HorizontalDiffusionMap(data, alpha=0.5, bandwidth=1.0, name="ct", n_neighbors=10)

P = hdm.transition_matrix           # row-stochastic Markov matrix
W, D_W = hdm.W, hdm.D_W             # the pieces needed to form T

walker = hdm.random_walk(idx=0)
path = [next(walker) for _ in range(100)]
```

`examples/connecticut.py` reproduces the numbers above from an ensemble file. The ensemble data is not included here.

## Acknowledgments

Thanks to Greg Herschlag and Biji Wong for leading the project, to Spencer Whitehead, who managed it and wrote the `districting-tools` library this code began in, and to my teammates Peakay Clifford, Eileen Santana, Felix Sesin, and Leo Yang.

## References

Gao, T. (2019). The diffusion geometry of fibre bundles: Horizontal diffusion maps. *Applied and Computational Harmonic Analysis*. https://doi.org/10.1016/j.acha.2019.08.001 ([arXiv:1602.02330](https://arxiv.org/abs/1602.02330))

Coifman, R. R., and Lafon, S. (2006). Diffusion maps. *Applied and Computational Harmonic Analysis*, 21(1), 5–30.
