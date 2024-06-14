# Generalized upwind summation-by-parts operators and their application to nodal discontinuous Galerkin methods

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11661785.svg)](https://doi.org/10.5281/zenodo.11661785)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@online{glaubitz2024generalized,
	title={Generalized upwind summation-by-parts operators and 
               their application to nodal discontinuous Galerkin methods},
  author={Glaubitz, Jan and Ranocha, Hendrik and 
          Winters, Andrew Ross and Schlottke-Lakemper, Michael and 
          {\"O}ffner, Philipp and Gassner, Gregor Josef},
  year={2024},
  month={06},
  doi={TODO},
  eprint={TODO},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{glaubitz2024generalizedRepro,
  title={Reproducibility repository for
         "{G}eneralized upwind summation-by-parts operators and 
         their application to nodal discontinuous Galerkin methods"},
  author={Glaubitz, Jan and Ranocha, Hendrik and 
          Winters, Andrew Ross and Schlottke-Lakemper, Michael and 
          {\"O}ffner, Philipp and Gassner, Gregor Josef},
  year={2024},
  howpublished={\url{https://github.com/trixi-framework/paper-2024-generalized-upwind-sbp}},
  doi={10.5281/zenodo.11661785}
}
```

## Abstract

There is a pressing demand for robust, high-order baseline schemes for conservation laws that minimize reliance on supplementary stabilization. 
In this work, we respond to this demand by developing new baseline schemes within a nodal discontinuous Galerkin (DG) framework, utilizing upwind summation-by-parts (USBP) operators and flux vector splittings. 
To this end, we demonstrate the existence of USBP operators on arbitrary grid points and provide a straightforward procedure for their construction. 
Our method encompasses a broader class of USBP operators, not limited to equidistant grid points.
This approach facilitates the development of novel USBP operators on Legendre--Gauss--Lobatto (LGL) points, which are suited for nodal discontinuous Galerkin (DG) methods. 
The resulting DG-USBP operators combine the strengths of traditional summation-by-parts (SBP) schemes with the benefits of upwind discretizations, including inherent dissipation mechanisms. 
Through numerical experiments, ranging from one-dimensional convergence tests to multi-dimensional curvilinear and under-resolved flow simulations, we find that DG-USBP operators, when integrated with flux vector splitting methods, foster more robust baseline schemes without excessive artificial dissipation.


## Numerical experiments

The numerical experiments presented in the paper use
[Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
To reproduce the numerical experiments using Trixi.jl, you need to install
[Julia](https://julialang.org/).

The subfolder `code` of this repository contains a `README.md` file with
instructions to reproduce the Cartesian mesh numerical experiments and
the subfolder `code_curved` contains a `README.md` file with instructions
to reproduce the curvilinear mesh numerical experiments.

The Cartesian mesh numerical experiments were carried out using Julia v1.9.4
and the curvilinear mesh results were carried out using Julia 1.10.0.


## Authors

- [Jan Glaubitz](https://www.janglaubitz.com) (MIT, USA)
- [Hendrik Ranocha](https://ranocha.de) (Johannes Gutenberg University Mainz, Germany)
- [Andrew Winters](https://liu.se/en/employee/andwi94) (Linköping University, Sweden)
- [Michael Schlottke-Lakemper](https://www.uni-augsburg.de/fakultaet/mntf/math/prof/hpsc) (University of Augsburg, Germany)
- [Philipp Öffner](https://philippoeffner.de) (TU Clausthal, Germany)
- [Gregor J. Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner/) (University of Cologne, Germany)


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
