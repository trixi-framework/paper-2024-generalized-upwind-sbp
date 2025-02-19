# Generalized upwind summation-by-parts operators and their application to nodal discontinuous Galerkin methods

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11661785.svg)](https://doi.org/10.5281/zenodo.11661785)

This repository contains information and code to reproduce the results presented in the
article
```bibtex
@article{glaubitz2025generalized,
  title={Generalized upwind summation-by-parts operators and their
         application to nodal discontinuous {G}alerkin methods},
  author={Glaubitz, Jan and Ranocha, Hendrik and Winters, Andrew Ross and
          Schlottke-Lakemper, Michael and {\"O}ffner, Philipp and
          Gassner, Gregor Josef},
  journal={Journal of Computational Physics},
  volume={529},
  pages={113841},
  year={2025},
  month={05},
  doi={10.1016/j.jcp.2025.113841},
  eprint={2406.14557},
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

High-order numerical methods for conservation laws are highly sought after due to
their potential efficiency. However, it is challenging to ensure their robustness,
particularly for under-resolved flows. Baseline high-order methods often incorporate
stabilization techniques that must be applied judiciously—sufficient to ensure
simulation stability but restrained enough to prevent excessive dissipation and
loss of resolution. Recent studies have demonstrated that combining upwind summation-by-parts (USBP)
operators with flux vector splitting can increase the robustness of finite difference (FD)
schemes without introducing excessive artificial dissipation. This work investigates whether
the same approach can be applied to nodal discontinuous Galerkin (DG) methods. To this end,
we demonstrate the existence of USBP operators on arbitrary grid points and provide
a straightforward procedure for their construction. Our discussion encompasses a
broad class of USBP operators, not limited to equidistant grid points, and enables
the development of novel USBP operators on Legendre–Gauss–Lobatto (LGL) points that are
well-suited for nodal DG methods. We then examine the robustness properties of the
resulting DG-USBP methods for challenging examples of the compressible Euler equations,
such as the Kelvin–Helmholtz instability. Similar to high-order FD-USBP schemes,
we find that combining flux vector splitting techniques with DG-USBP operators
does not lead to excessive artificial dissipation. Furthermore, we find that
combining lower-order DG-USBP operators on three LGL points with flux vector splitting
indeed increases the robustness of nodal DG methods. However, we also observe that
higher-order USBP operators offer less improvement in robustness for DG methods
compared to FD schemes. We provide evidence that this can be attributed to USBP methods
adding dissipation only to unresolved modes, as FD schemes typically have more
unresolved modes than nodal DG methods.


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
