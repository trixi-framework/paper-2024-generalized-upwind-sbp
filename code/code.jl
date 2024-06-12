# Install dependencies
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load dependencies
using LinearAlgebra
using Statistics
using DelimitedFiles: DelimitedFiles, writedlm, readdlm

using Trixi
using OrdinaryDiffEq
using DiffEqCallbacks
using SummationByPartsOperators

import PolyesterWeave, ThreadingUtilities

using LaTeXStrings
using Plots: Plots, plot, plot!, scatter, scatter!, savefig
Plots.pyplot()
using Trixi2Vtk: trixi2vtk
using PrettyTables: PrettyTables, pretty_table, ft_printf

const figdir = joinpath(dirname(@__DIR__), "figures")


### Print the USBP operators ###

function print_USBP_operators(; N = 3, σ = -10^(-1) )

    # Note: The points, weights, and Vandermonde matrices below were computed in a separate Matlab script
    if N == 3
        # Gauss-Lobatto nodes
        nodes = [-1.0, 0.0, 1.0]
        # Gauss-Lobatto weights
        weights = [1/3, 4/3, 1/3]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = ( 1/sqrt(6) ) * [
            sqrt(2) -sqrt(3) 1
            sqrt(2) 0 -2
            sqrt(2) sqrt(3) 1
        ]

    elseif N == 4
        # Gauss-Lobatto nodes
        nodes = [-1, -1/sqrt(5), 1/sqrt(5), 1]
        # Gauss-Lobatto weights
        weights = [1/6, 5/6, 5/6, 1/6]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.500000000000000000 -0.645497224367902800 0.499999999999999889 -0.288675134594812921
            0.500000000000000000 -0.288675134594812866 -0.500000000000000000 0.645497224367902800
            0.500000000000000000 0.288675134594812866 -0.500000000000000000 -0.645497224367902800
            0.500000000000000000 0.645497224367902800 0.499999999999999889 0.288675134594812921
        ]

    elseif N == 5
        # Gauss-Lobatto nodes
        nodes = [-1, -sqrt(3/7), 0, sqrt(3/7), 1]
        # Gauss-Lobatto weights
        weights = [1/10, 49/90, 32/45, 49/90, 1/10]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.447213595499957928 -0.591607978309961591 0.500000000000000000 -0.387298334620741758 0.223606797749978908
            0.447213595499957928 -0.387298334620741702 -0.166666666666666685 0.591607978309961591 -0.521749194749950962
            0.447213595499957928 0.000000000000000000 -0.666666666666666741 0.000000000000000000 0.596284793999943941
            0.447213595499957928 0.387298334620741702 -0.166666666666666685 -0.591607978309961591 -0.521749194749950962
            0.447213595499957928 0.591607978309961591 0.500000000000000000 0.387298334620741758 0.223606797749978908
        ]

    elseif N == 6
        # Gauss-Lobatto nodes
        nodes = [
            -1, -sqrt( ( 7+2*sqrt(7) ) / 21 ), -sqrt( ( 7-2*sqrt(7) ) / 21 ),
            sqrt( ( 7-2*sqrt(7) ) / 21 ), sqrt( ( 7+2*sqrt(7) ) / 21 ), 1
        ]
        # Gauss-Lobatto weights
        weights = [
            1/15, (14-sqrt(7))/30, (14+sqrt(7))/30,
            (14+sqrt(7))/30, (14-sqrt(7))/30, 1/15
        ]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.408248290463863073 -0.547722557505166074 0.483045891539647831 -0.408248290463863017 0.316227766016837830 -0.182574185835055497
            0.408248290463863073 -0.419038058655589740 0.032338332982759031 0.367654222400928044 -0.576443896275456780 0.435014342463467985
            0.408248290463863073 -0.156227735687855862 -0.515384224522407175 0.445155822251155409 0.260216130258618783 -0.526715472069829382
            0.408248290463863073 0.156227735687855862 -0.515384224522407175 -0.445155822251155409 0.260216130258618783 0.526715472069829382
            0.408248290463863073 0.419038058655589740 0.032338332982759031 -0.367654222400928044 -0.576443896275456780 -0.435014342463467985
            0.408248290463863073 0.547722557505166074 0.483045891539647831 0.408248290463863017 0.316227766016837830 0.182574185835055497
        ]

    end

    # Dissipation matrix
    σ_vector = zeros(N)
    σ_vector[N] = σ
    S = U*diagm(σ_vector)*transpose(U);
    # Inverse norm matrix
    P_inv = diagm( 1 ./ weights )
    # Central and upwind SBP operators
    Dc = Matrix(legendre_derivative_operator(-1.0, 1.0, N))
    Dm = Dc - P_inv*S
    Dp = Dc + P_inv*S
    xmin = -1.0
    xmax = +1.0
    D_upw = UpwindOperators(
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dm, 0, nothing),
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dc, 0, nothing),
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dp, 0, nothing)
    )

    # print the nodes
    @info "DG-USBP nodes x on [-1,1] for N=$(N)"
    @show nodes

    # print the weights
    @info "DG-USBP weights p on [-1,1] for N=$(N)"
    @show weights

    # print the central SBP operator
    @info "Central degree $(N-1) SBP operator D on [-1,1] for N=$(N)"
    @show Dc

    # print the DOP Vandermonde matrix
    @info "DOP Vandermonde matrix V on [-1,1] for N=$(N)"
    @show U

    # print the dissipation matrix S
    @info "Dissipation matrix S = V Σ Vᵀ for Σ = diag(0,...,0,$(σ)) and N=$(N)"
    @show U

    # print the USBP operators D_{\pm}
    @info "USBP operators D₊ = D + P⁻¹S/2 for N=$(N)"
    @show Dp

    # print the USBP operators D_{\pm}
    @info "USBP operators D₋ = D - P⁻¹S/2 for N=$(N)"
    @show Dm

    return nothing
end


### Convergence experiments for 1d linear advection equation ###

function convergence_tests_1d_advection(; N = 3, latex = false)

    @info "1D linear advection, DG-USBP"
    refinement_levels = 1:7
    σ_values = [ -10^(-3), -10^(-2), -10^(-1), -10^(0) ]

    # USBP-DG method
    for σ in σ_values
        @info  "DG-USBP" σ
        _convergence_tests_1d_advection(; N, σ, refinement_levels, latex )
    end

    return nothing

end

function _convergence_tests_1d_advection(; N, σ, refinement_levels, latex = false )

    num_elements = Vector{Int}()
    errors = Vector{Float64}()

    for initial_refinement_level in refinement_levels
        nelements = 2^initial_refinement_level
        tol = 1.0e-12
        res = compute_errors_1d_advection(; N, σ, initial_refinement_level, tol )
        push!(num_elements, nelements)
        push!(errors, first(res.l2))
    end

    eoc = compute_eoc(num_elements, errors)

    # print results
    data = hcat(num_elements, errors, eoc)
    header = ["#elements", "L2 error", "L2 EOC"]
    kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                    ft_printf("%.2e", [2]),
                                    ft_printf("%.2f", [3])))
    pretty_table(data; kwargs...)
    if latex
        pretty_table(data; kwargs..., backend=Val(:latex))
    end

    return nothing
end

function compute_errors_1d_advection(; N, σ, initial_refinement_level, tol)
    equations = LinearScalarAdvectionEquation1D(1.0)

    function initial_condition(x, t, equations::LinearScalarAdvectionEquation1D)
        return SVector(sinpi(x[1] - equations.advection_velocity[1] * t))
    end

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

    flux_splitting = splitting_lax_friedrichs
    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralUpwind(flux_splitting),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    coordinates_min = -1.0
    coordinates_max =  1.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max=10_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    ode = semidiscretize(semi, (0.0, 5.0))
    sol = solve(ode, RDPK3SpFSAL49(); ode_default_options()...,
                abstol = tol, reltol = tol)

    analysis_callback = AnalysisCallback(semi)
    return analysis_callback(sol)
end

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) / log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end


### Convergence experiments for 1d compressible Euler equations ###

function convergence_tests_1d_euler(; N = 3, latex = false)

    @info "1D compressible Euler equations"
    refinement_levels = 1:7
    σ_values = [ -10^(-3), -10^(-2), -10^(-1), -10^(0) ]

    # flux differencing DGSEM
    @info  "DGSEM"
    for volume_flux in ( flux_ranocha, flux_shima_etal )
        @info "DGSEM" volume_flux
        _convergence_tests_1d_euler_DGSEM(; N, refinement_levels, volume_flux, latex )
    end

    # USBP-DG method
    for splitting in ( splitting_vanleer_haenel, splitting_steger_warming )
        @info  "DG-USBP" splitting
        for σ in σ_values
            @info  "DG-USBP" splitting σ
                _convergence_tests_1d_euler_USBP(; N, σ, refinement_levels, splitting, latex)
        end
    end

    return nothing

end

function _convergence_tests_1d_euler_DGSEM(; N, refinement_levels, volume_flux, latex = false)

    num_elements = Vector{Int}()
    errors = Vector{Float64}()

    for initial_refinement_level in refinement_levels
        nelements = 2^initial_refinement_level
        tol = 1.0e-13
        res = compute_errors_1d_euler_DGSEM(; N, initial_refinement_level, tol, volume_flux )
        push!(num_elements, nelements)
        push!(errors, first(res.l2))
    end

    eoc = compute_eoc(num_elements, errors)

    # print results
    data = hcat(num_elements, errors, eoc)
    header = ["#elements", "L2 error", "L2 EOC"]
    kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                    ft_printf("%.2e", [2]),
                                    ft_printf("%.2f", [3])))
    pretty_table(data; kwargs...)
    if latex
        pretty_table(data; kwargs..., backend=Val(:latex))
    end

    return nothing
end

function _convergence_tests_1d_euler_USBP(; N, σ, refinement_levels, splitting, latex = false )
    num_elements = Vector{Int}()
    errors = Vector{Float64}()

    for initial_refinement_level in refinement_levels
        nelements = 2^initial_refinement_level
        tol = 1.0e-13
        res = compute_errors_1d_euler_USBP(; N, σ, initial_refinement_level, tol, splitting )
        push!(num_elements, nelements)
        push!(errors, first(res.l2))
    end

    eoc = compute_eoc(num_elements, errors)

    # print results
    data = hcat(num_elements, errors, eoc)
    header = ["#elements", "L2 error", "L2 EOC"]
    kwargs = (; header, formatters=(ft_printf("%3d", [1]),
                                    ft_printf("%.2e", [2]),
                                    ft_printf("%.2f", [3])))
    pretty_table(data; kwargs...)
    if latex
        pretty_table(data; kwargs..., backend=Val(:latex))
    end

    return nothing
end

function compute_errors_1d_euler_DGSEM(; N, initial_refinement_level, tol, volume_flux )
    equations = CompressibleEulerEquations1D(1.4)

    initial_condition = initial_condition_convergence_test
    source_terms = source_terms_convergence_test

    # Use DGSEM
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
    #solver = DGSEM(; polydeg=N-1, surface_flux = flux_lax_friedrichs, volume_integral )
    solver = DGSEM(; polydeg=N-1, surface_flux = flux_hll, volume_integral )
    #solver = DGSEM(; polydeg=N-1, surface_flux = flux_hllc, volume_integral )

    coordinates_min = 0.0
    coordinates_max = 2.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 10_000)

    semi = SemidiscretizationHyperbolic( mesh, equations, initial_condition,
                                        solver; source_terms )

    ode = semidiscretize(semi, (0.0, 2.0))
    sol = solve(ode, RDPK3SpFSAL49(); ode_default_options()...,
                abstol = tol, reltol = tol)

    analysis_callback = AnalysisCallback(semi)

    return analysis_callback(sol)

end

function compute_errors_1d_euler_USBP(; N, σ, initial_refinement_level, tol, splitting )
    equations = CompressibleEulerEquations1D(1.4)

    initial_condition = initial_condition_convergence_test
    source_terms = source_terms_convergence_test

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralUpwind(splitting),
                   volume_integral = VolumeIntegralUpwind(splitting))

    coordinates_min = 0.0
    coordinates_max = 2.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 10_000)

    semi = SemidiscretizationHyperbolic( mesh, equations, initial_condition,
                                        solver; source_terms )

    ode = semidiscretize(semi, (0.0, 2.0))
    sol = solve(ode, RDPK3SpFSAL49(); ode_default_options()...,
                abstol = tol, reltol = tol)

    analysis_callback = AnalysisCallback(semi)

    return analysis_callback(sol)

end


### Analysis of the spectra ###

function plot_spectra()

    ## Compare DG-USBP and DG-SBP, 8 elements
    let refinement_parameter = 4
        for N = 3:5

            fig = plot(xguide = "Real part", yguide = "Imaginary part")

            # Classical DGSEM
            let polydeg = N-1
                initial_refinement_level = refinement_parameter
                λ = compute_spectrum_1d_DGSEM(; initial_refinement_level, polydeg )
                @show extrema(real, λ)
                @show maximum(abs, λ)
                λ = sort_spectrum(λ)
                label = "DG-SBP with $(2^initial_refinement_level) elements and degree $polydeg"
                plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :solid)
            end

            # USBP operators

            let
                σ = -10^(-2)
                initial_refinement_level = refinement_parameter
                λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
                @show extrema(real, λ)
                @show maximum(abs, λ)
                λ = sort_spectrum(λ)
                label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-10⁻²"
                plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dash)
            end

            let
                σ = -10^(-1)
                initial_refinement_level = refinement_parameter
                λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
                @show extrema(real, λ)
                @show maximum(abs, λ)
                λ = sort_spectrum(λ)
                label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-10⁻¹"
                plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dot)
            end

            let
                σ = -1
                initial_refinement_level = refinement_parameter
                λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
                @show extrema(real, λ)
                @show maximum(abs, λ)
                λ = sort_spectrum(λ)
                label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-1"
                plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dashdot)
            end
            plot!(fig, legend = :outertop)
            savefig(fig, joinpath(figdir,
                "spectra_N$(N)_USBP_vs_DGSEM_$(2^refinement_parameter)elements.pdf"))

        end
    end

    # Compare degree one DG-USBP and FD-SBP with accuracy 4
    # Degree one (d=1) DG-USBP operators on three (N=3) Gauss-Lobatto points
    accuracy_order = 4
    let N = 3

        initial_refinement_level = 5
        fig = plot(xguide = "Real part", yguide = "Imaginary part")

        # Degree one (d=1) DG-USBP operators on three (N=3) Gauss-Lobatto points
        let
            σ = -10^(-3)
            λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-10⁻³"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :solid)
        end

        let
            σ = -1
            λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-1"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dot)
        end

        # FD-USBP operators
        let
            nnodes = 24
            initial_refinement_level = 2
            λ = compute_spectrum_1d_FD(; initial_refinement_level, nnodes, accuracy_order)
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "FD-USBP with $(2^initial_refinement_level) elements and $(nnodes) nodes"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dash)
        end

        let
            nnodes = 12
            initial_refinement_level = 3
            λ = compute_spectrum_1d_FD(; initial_refinement_level, nnodes, accuracy_order)
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "FD-USBP with $(2^initial_refinement_level) elements and $(nnodes) nodes"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dashdot)
        end

        plot!(fig, legend = :outertop)
        savefig(fig, joinpath(figdir,
            "spectra_N$(N)_USBP_vs_FD_accuracy$(accuracy_order).pdf"))

    end

    # Compare degree one DG-USBP and FD-SBP with accuracy 4
    # Degree two (d=2) DG-USBP operators on four (N=4) Gauss-Lobatto points
    accuracy_order = 4
    let N = 4

        initial_refinement_level = 5
        fig = plot(xguide = "Real part", yguide = "Imaginary part")

        let
            σ = -10^(-3)
            λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-10⁻³"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :solid)
        end

        let
            σ = -1
            λ = compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "DG-USBP with $(2^initial_refinement_level) elements and λ=-1"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dot)
        end

        # FD-USBP operators
        let
            nnodes = 32
            initial_refinement_level = 2
            λ = compute_spectrum_1d_FD(; initial_refinement_level, nnodes, accuracy_order)
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "FD-USBP with $(2^initial_refinement_level) elements and $(nnodes) nodes"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dash)
        end

        let
            nnodes = 16
            initial_refinement_level = 3
            λ = compute_spectrum_1d_FD(; initial_refinement_level, nnodes, accuracy_order)
            @show extrema(real, λ)
            @show maximum(abs, λ)
            λ = sort_spectrum(λ)
            label = "FD-USBP with $(2^initial_refinement_level) elements and $(nnodes) nodes"
            plot!(fig, real.(λ), imag.(λ); label, plot_kwargs()..., linewidth = 2, linestyle = :dashdot)
        end

        plot!(fig, legend = :outertop)
        savefig(fig, joinpath(figdir,
            "spectra_N$(N)_USBP_vs_FD_accuracy$(accuracy_order).pdf"))

    end

    @info "1D spectra saved in the directory `figdir`" figdir
        return nothing

end

function plot_kwargs()
    fontsizes = (
        xtickfontsize = 12, ytickfontsize = 12,
        xguidefontsize = 12, yguidefontsize = 12,
        legendfontsize = 12)
    (; linewidth = 3, gridlinewidth = 2,
        markersize = 8, markerstrokewidth = 4,
        fontsizes...)
end

function sort_spectrum(λ)
    idx_pos = imag.(λ) .> 0
    pos = λ[idx_pos]
    neg = λ[.!(idx_pos)]
    sort!(pos; lt = !isless, by = real)
    sort!(neg; lt = isless, by = real)
    return vcat(pos, neg)
end

function compute_spectrum_1d_USBP(; initial_refinement_level, N, σ )
    equations = LinearScalarAdvectionEquation1D(1.0)

    function initial_condition(x, t, equations::LinearScalarAdvectionEquation1D)
        return SVector(sinpi(x[1] - equations.advection_velocity[1] * t))
    end

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

    flux_splitting = splitting_lax_friedrichs
    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralUpwind(flux_splitting),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    coordinates_min = -1.0
    coordinates_max =  1.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max=10_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    J = jacobian_ad_forward(semi)
    λ = eigvals(J)
    return λ
end

function compute_spectrum_1d_DGSEM(; initial_refinement_level, polydeg)
    equations = LinearScalarAdvectionEquation1D(1.0)

    function initial_condition(x, t, equations::LinearScalarAdvectionEquation1D)
        return SVector(sinpi(x[1] - equations.advection_velocity[1] * t))
    end

    solver = DGSEM(polydeg = polydeg, surface_flux = flux_godunov)

    coordinates_min = -1.0
    coordinates_max =  1.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max=10_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    J = jacobian_ad_forward(semi)
    λ = eigvals(J)
    return λ
end

function compute_spectrum_1d_FD(; initial_refinement_level, nnodes, accuracy_order)
    equations = LinearScalarAdvectionEquation1D(1.0)

    function initial_condition(x, t, equations::LinearScalarAdvectionEquation1D)
    return SVector(sinpi(x[1] - equations.advection_velocity[1] * t))
    end

    D_upw = upwind_operators( SummationByPartsOperators.Mattsson2017;
        derivative_order = 1,
        accuracy_order,
        xmin = -1.0, xmax = 1.0,
        N = nnodes
    )
    flux_splitting = splitting_lax_friedrichs
    solver = FDSBP(D_upw,
    surface_integral = SurfaceIntegralUpwind(flux_splitting),
    volume_integral = VolumeIntegralUpwind(flux_splitting))

    coordinates_min = -1.0
    coordinates_max =  1.0
    mesh = TreeMesh(coordinates_min, coordinates_max;
    initial_refinement_level,
    n_cells_max=10_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    J = jacobian_ad_forward(semi)
    λ = eigvals(J)
    return λ
end


### Local linear/energy experiments ###

function local_linear_stability(; N = 4, latex = false)

    @info "Full upwind discretization (only D_−) for" N
    refinement_levels = 1:5
    σ_exponents = [ -Inf, -2, 0 ]

    num_elements = Int[]
    values_sigma = Int[]
    max_real_parts = zeros(Float64,length(refinement_levels),length(σ_exponents))
    #max_real_parts = Float64[]

    i = 0
    for refinement_parameter in refinement_levels

        i = i+1
        nelements = 2^refinement_parameter
        push!(num_elements, nelements)

        j = 0
        for σ_exp in σ_exponents
            j = j+1
            σ = -exp(σ_exp)
            λ = compute_spectrum_burgers_DG_USBP_full(; N, nelements, σ )
            max_real_parts[i,j] = maximum(real, λ)
        end

    end

    # print results
    data = hcat( num_elements, max_real_parts )
    header = ["J",
        "max. real parts for log(-σ)=-Inf",
        "max. real parts for log(-σ)=-2",
        "max. real parts for log(-σ)=0"
    ]
    kwargs = (; header, formatters=( ft_printf("%2d", [1]), ft_printf("%9.1e", [2]),
        ft_printf("%9.1e", [3]), ft_printf("%9.1e", [4])
    ))
    pretty_table(data; kwargs...)
    if latex
        pretty_table(data; kwargs..., backend=Val(:latex))
    end

    return nothing
end


function compute_spectrum_burgers_DG_USBP_full(; N, nelements, σ )

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

    D_local = D_upw.minus

    mesh = UniformPeriodicMesh1D(xmin = -1.0, xmax = 1.0, Nx = nelements)

    D = couple_discontinuously(D_local, mesh, Val(:minus))

    u0 = rand(size(D, 2))

    J = Trixi.ForwardDiff.jacobian(u0) do u
        -D * (u.^2 ./ 2)
    end

    return eigvals(J)

end


### Experiments for the isentropic vortex ### 

function experiments_isentropic_vortex()
     
    refinement_levels = 4:5
    values_sigma = [-10^(-3),-10^(-1)] 
    for σ in values_sigma
        for initial_refinement_level in refinement_levels
            isentropic_vortex_generate_data( initial_refinement_level, σ )
            isentropic_vortex_plot_results( initial_refinement_level, σ )
        end
    end

end

function isentropic_vortex_generate_data( initial_refinement_level, σ ) 
    
    i = Int( -log10(-σ) )
    # DG-USBP
    for N in 3:6 
        
        @info "Generate data for DG-USBP for N=$(N), $(4^initial_refinement_level) elements, and σ=$(σ)" 
        t, error_density = compute_error_isentropic_vortex(; 
            N, σ, initial_refinement_level
        )
        open(joinpath(figdir, "isentropic_vortex_N$(N)_DGUSBP_$(4^initial_refinement_level)elements_sigma$(i).dat"), "w") do io
            println(io, "# t\tL2_error_density")
            writedlm(io, hcat(t, error_density)
        )
        end

    end

end

function isentropic_vortex_plot_results( initial_refinement_level, σ )
    
    fig = plot(xguide = L"Time $t$", yguide = L"$L^2$ error of the density";
               xscale = :log10, yscale = :log10,
               plot_kwargs()...)

    linestyles = [:dash, :dot, :dashdot, :solid]
    
    i = Int( -log10(-σ) ) 

    # DG-USBP results
    for (N, linestyle) in zip(3:6, linestyles)
        data = readdlm(joinpath(figdir, 
            "isentropic_vortex_N$(N)_DGUSBP_$(4^initial_refinement_level)elements_sigma$(i).dat"), 
            comments = true
        )
        plot!(fig, data[:, 1], data[:, 2];
              label = "N = $(N)", linestyle,
              plot_kwargs()...)
    end

    plot!(fig, legend = :bottomright)
    savefig(fig, joinpath(figdir, "isentropic_vortex_$(4^initial_refinement_level)elements_sigma$(i).pdf"))
    @info "Error plot saved in the directory `figdir`" figdir
    return nothing

end

function compute_error_isentropic_vortex(; N, σ, initial_refinement_level,
                                           flux_splitting = splitting_steger_warming,
                                           volume_flux = flux_ranocha_turbo,
                                           surface_flux = flux_lax_friedrichs,
                                           tspan = (0.0, 1000.0),
                                           tol = 1.0e-6)
    equations = CompressibleEulerEquations2D(1.4)

    """
        initial_condition_isentropic_vortex(x, t, equations)

    The classical isentropic vortex test case of
    - Chi-Wang Shu (1997)
    Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
    Schemes for Hyperbolic Conservation Laws.
    [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
    """
    function initial_condition(x, t, equations::CompressibleEulerEquations2D)
        ϱ0 = 1.0               # background density
        v0 = SVector(1.0, 1.0) # background velocity
        p0 = 10.0              # background pressure
        ε = 10.0               # vortex strength
        L = 10.0               # size of the domain per coordinate direction

        T0 = p0 / ϱ0           # background temperature
        γ = equations.gamma    # ideal gas constant

        vortex_center(x, L) = mod(x + L/2, L) - L/2
        x0 = v0 * t            # current center of the vortex
        dx = vortex_center.(x - x0, L)
        r2 = sum(abs2, dx)

        # perturbed primitive variables
        T = T0 - (γ - 1) * ε^2 / (8 * γ * π^2) * exp(1 - r2)
        v = v0 + ε / (2 * π) * exp(0.5 * (1 - r2)) * SVector(-dx[2], dx[1])
        ϱ = ϱ0 * (T / T0)^(1 / (γ - 1))
        p = ϱ * T

        return prim2cons(SVector(ϱ, v..., p), equations)
    end

    if σ === nothing
        # Use DGSEM
        volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
        solver = DGSEM(; polydeg=3, surface_flux, volume_integral)
    else 
        # Use DG-USBP 
        # Compute the DG-USBP operators 
        D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

        solver = FDSBP(D_upw,
        surface_integral = SurfaceIntegralUpwind(flux_splitting),
        volume_integral = VolumeIntegralUpwind(flux_splitting)
        )

    end

    coordinates_min = (-5.0, -5.0)
    coordinates_max = ( 5.0,  5.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 100_000,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    ode = semidiscretize(semi, tspan)

    saveat = range(tspan..., step = 20)[2:end]
    saved_values = SavedValues(Float64, Float64)
    save_func = let cb = AnalysisCallback(semi)
        function save_func(u_ode, t, integrator)
            semi = integrator.p
            analysis_callback = cb.affect!
            (; analyzer) = analysis_callback
            cache_analysis = analysis_callback.cache

            l2_error, linf_error = Trixi.calc_error_norms(u_ode, t,
                                                          analyzer, semi,
                                                          cache_analysis)
            return first(l2_error)
        end
    end
    saving = SavingCallback(save_func, saved_values; saveat)

    sol = solve(ode, RDPK3SpFSAL49();
                abstol = tol, reltol = tol,
                ode_default_options()..., callback = saving, tstops = saveat)
    return (; t = saved_values.t, error_density = saved_values.saveval)
end


### Kelvin-Helmholtz instability ###

function experiments_kelvin_helmholtz_instability(; N = 4, latex = false)
    initial_refinement_levels = 2:6

    # Upwind SBP operators
    @info "DG-USBP operators"
    σ_values = [ -10^(-3), -10^(-2), -10^(-1), -10^(0) ]
    for flux_splitting in (splitting_vanleer_haenel, splitting_steger_warming)
        #@info flux_splitting
        final_times = Matrix{Float64}(undef, length(σ_values), length(initial_refinement_levels) )

        for (i, σ) in enumerate(σ_values)
            for (j, initial_refinement_level) in enumerate(initial_refinement_levels)
                t = blowup_kelvin_helmholtz(; N, σ, initial_refinement_level, flux_splitting)
                final_times[i, j] = t
            end
        end

        # print results
        data = hcat( σ_values, final_times)
        header = vcat(0, 4 .^ initial_refinement_levels )
        kwargs = (; header, title = "DG-USBP" * string(flux_splitting),
                    formatters=(ft_printf("%.4f", [1]),
                                ft_printf("%.2f", 2:size(data,2))))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
        println()
    end

    # DGSEM
    @info "DGSEM"
    polydeg = N-1
    for volume_flux in (flux_ranocha, flux_shima_etal)
        #@info volume_flux
        final_times = Matrix{Float64}(undef, 1, length(initial_refinement_levels) )

        for (j, initial_refinement_level) in enumerate(initial_refinement_levels)
            t = blowup_kelvin_helmholtz(; N, σ = 0, polydeg,
                initial_refinement_level, volume_flux)
            final_times[1, j] = t
        end

        # print results
        data = hcat( polydeg, final_times)
        header = vcat(0, 4 .^ initial_refinement_levels )
        kwargs = (; header, title = "DGSEM" * string(volume_flux),
                    formatters=(ft_printf("%3d", [1]),
                                ft_printf("%.2f", 2:size(data,2))))
        pretty_table(data; kwargs...)
        if latex
            pretty_table(data; kwargs..., backend=Val(:latex))
        end
        println()
    end

 return nothing
end

function blowup_kelvin_helmholtz(; N, σ, initial_refinement_level = 2,
                                   flux_splitting = splitting_vanleer_haenel,
                                   polydeg = nothing,
                                   volume_flux = flux_ranocha,
                                   tol = 1.0e-6)
    equations = CompressibleEulerEquations2D(1.4)

    function initial_condition(x, t, equations::CompressibleEulerEquations2D)
        # change discontinuity to tanh
        # typical resolution 128^2, 256^2
        # domain size is [-1,+1]^2
        slope = 15
        B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
        rho = 0.5 + 0.75 * B
        v1 = 0.5 * (B - 1)
        v2 = 0.1 * sin(2 * pi * x[1])
        p = 1.0
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    if polydeg === nothing

        # Use upwind SBP discretization
        D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

        solver = FDSBP(D_upw,
                       surface_integral = SurfaceIntegralUpwind(flux_splitting),
                       volume_integral = VolumeIntegralUpwind(flux_splitting))

        #@info "Kelvin-Helmholtz instability" initial_refinement_level flux_splitting
    else
        # Use DGSEM
        volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
        solver = DGSEM(; polydeg, surface_flux = flux_lax_friedrichs, volume_integral)

        #@info "Kelvin-Helmholtz instability" polydeg initial_refinement_level volume_flux
    end

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 100_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    @show Trixi.ndofs(semi)

    tspan = (0.0, 15.0)
    ode = semidiscretize(semi, tspan)

    integrator = init(ode, SSPRK43(); controller = PIDController(0.55, -0.27, 0.05),
                      abstol = tol, reltol = tol,
                      ode_default_options()...)

    try
        solve!(integrator)
    catch error
        @info "Blow-up" integrator.t
        reset_threads!()
    end

    return integrator.t
end

# https://github.com/JuliaSIMD/Polyester.jl/issues/30
function reset_threads!()
    PolyesterWeave.reset_workers!()
    for i in 1:(Threads.nthreads() - 1)
        ThreadingUtilities.initialize_task(i)
    end
    return nothing
end

function run_kelvin_helmholtz(; N, σ, initial_refinement_level = 2,
                                flux_splitting = splitting_vanleer_haenel,
                                source_of_coefficients = Mattsson2017,
                                polydeg = nothing,
                                volume_flux = flux_ranocha_turbo,
                                tol = 1.0e-6)
    equations = CompressibleEulerEquations2D(1.4)

    function initial_condition(x, t, equations::CompressibleEulerEquations2D)
        # change discontinuity to tanh
        # typical resolution 128^2, 256^2
        # domain size is [-1,+1]^2
        slope = 15
        B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
        rho = 0.5 + 0.75 * B
        v1 = 0.5 * (B - 1)
        v2 = 0.1 * sin(2 * pi * x[1])
        p = 1.0
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    if polydeg === nothing
        # Use upwind SBP discretization
        D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator

        solver = FDSBP(D_upw,
                       surface_integral=SurfaceIntegralUpwind(flux_splitting),
                    #    surface_integral=SurfaceIntegralStrongForm(flux_lax_friedrichs),
                       volume_integral=VolumeIntegralUpwind(flux_splitting))
    else
        # Use DGSEM
        volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
        solver = DGSEM(; polydeg, surface_flux = flux_lax_friedrichs, volume_integral)
    end

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 100_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    tspan = (0.0, 15.0)
    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_interval = 1000
    analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

    alive_callback = AliveCallback(analysis_interval=analysis_interval)

    saving_callback = SaveSolutionCallback(; interval = 100,
                                             save_final_solution = true,
                                             output_directory = joinpath(@__DIR__, "out_dev"),
                                            #  solution_variables = cons2cons)
                                             solution_variables = cons2prim)

    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback,
                            saving_callback)

    integrator = init(ode, SSPRK43(); controller = PIDController(0.55, -0.27, 0.05),
                      abstol = tol, reltol = tol,
                      ode_default_options()..., callback=callbacks)
    try
        solve!(integrator)
    catch err
        @warn "Crashed at time" integrator.t
        saving_callback.affect!(integrator)
        reset_threads!()
    end
    summary_callback() # print the timer summary

    return nothing
end


### Inviscid Taylor-Green vortex ###

function dissipation_experiments_taylor_green_vortex(; N = 3, refinement_level = 3, σ = -10^(-3) )

    # Run simulations and save analysis data
    initial_refinement_level = refinement_level

    # DG-USBP
    filename = "analysis_TGV_USBP_level$(initial_refinement_level)_N$(N).dat"
    run_taylor_green(; N, σ, polydeg = nothing,
        initial_refinement_level, analysis_interval = 10,
        analysis_filename = joinpath(@__DIR__, filename)
    )

    # DGSEM
    filename = "analysis_TGV_DGSEM_level$(initial_refinement_level)_N$(N).dat"
    run_taylor_green(; N, σ = 0, polydeg = N-1,
        initial_refinement_level, analysis_interval = 10,
        analysis_filename = joinpath(@__DIR__, filename)
    )

    # Plot results
    fig_kinetic_energy = plot(xguide = L"Time $t$", yguide = L"Kinetic energy $E_\mathrm{kin}$")
    fig_dissipation_rate = plot(xguide = L"Time $t$", yguide = L"Dissipation rate $-\Delta E_\mathrm{kin} / \Delta t$")

    kwargs = (; label = "DG-USBP", plot_kwargs()...)
    data = readdlm(joinpath(@__DIR__, "analysis_TGV_USBP_level$(initial_refinement_level)_N$(N).dat"), comments = true)
    time = data[:, 2]
    kinetic_energy = data[:, 16]
    plot!(fig_kinetic_energy, time, kinetic_energy; kwargs...)
    plot!(fig_dissipation_rate, dissipation_rate(time, kinetic_energy)...; kwargs...)

    kwargs = (; label = "DGSEM", linestyle = :dot, plot_kwargs()...)
    data = readdlm(joinpath(@__DIR__, "analysis_TGV_DGSEM_level$(initial_refinement_level)_N$(N).dat"), comments = true)
    time = data[:, 2]
    kinetic_energy = data[:, 16]
    plot!(fig_kinetic_energy, time, kinetic_energy; kwargs...)
    plot!(fig_dissipation_rate, dissipation_rate(time, kinetic_energy)...; kwargs...)

    savefig(fig_kinetic_energy, joinpath(figdir, "TGV_kinetic_energy_level$(initial_refinement_level)_N$(N).pdf"))
    savefig(fig_dissipation_rate, joinpath(figdir, "TGV_dissipation_rate_level$(initial_refinement_level)_N$(N).pdf"))

    @info "Kinetic energy plots saved in the directory `figdir`" figdir
    return nothing
end

function dissipation_rate(time, kinetic_energy)
    dissipation_rate = zeros(length(time) - 2)
    for i in eachindex(dissipation_rate)
        dissipation_rate[i] = - (kinetic_energy[i + 2] - kinetic_energy[i]) / (time[i + 2] - time[i])
    end
    return time[(begin + 1):(end - 1)], dissipation_rate
end

function run_taylor_green(; N = 3, σ = 0, polydeg = nothing,
                            initial_refinement_level = 1,
                            flux_splitting = splitting_steger_warming,
                            volume_flux = flux_ranocha_turbo,
                            surface_flux = flux_lax_friedrichs,
                            tol = 1.0e-6,
                            analysis_interval = 10000,
                            analysis_filename = "analysis.dat",
                            Mach = 0.1,
                            tspan = (0.0, 20.0))
    equations = CompressibleEulerEquations3D(1.4)

    function initial_condition(x, t, equations::CompressibleEulerEquations3D)
        A  = 1.0 # magnitude of speed
        Ms = Mach # maximum Mach number

        rho = 1.0
        v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
        v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
        v3  = 0.0
        p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
        p   = p + 1.0/16.0 * A^2 * rho * (cos(2 * x[1]) * cos(2 * x[3]) +
                                            2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
                                            cos(2 * x[2]) * cos(2 * x[3]))

        return prim2cons(SVector(rho, v1, v2, v3, p), equations)
    end

    if polydeg === nothing

        # Use upwind SBP discretization
        D_upw = compute_DG_UpwindOperators( N, σ ) # construct USBP operator
        solver = FDSBP(D_upw,
                       surface_integral = SurfaceIntegralUpwind(flux_splitting),
                       volume_integral = VolumeIntegralUpwind(flux_splitting))

    else
        # Use DGSEM
        volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
        solver = DGSEM(; polydeg, surface_flux, volume_integral)

    end

    coordinates_min = (-1.0, -1.0, -1.0) .* pi
    coordinates_max = ( 1.0,  1.0,  1.0) .* pi
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level,
                    n_cells_max = 8 ^ (initial_refinement_level + 1))

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()

    analysis_callback = AnalysisCallback(semi; interval = analysis_interval,
                                         save_analysis = true,
                                         analysis_filename,
                                         extra_analysis_integrals=(energy_total,
                                                                   energy_kinetic,
                                                                   energy_internal,))

    alive_callback = AliveCallback(analysis_interval=analysis_interval)

    callbacks = CallbackSet(summary_callback,
                            analysis_callback,
                            alive_callback)

    sol = solve(ode, SSPRK43(); controller = PIDController(0.55, -0.27, 0.05),
                abstol = tol, reltol = tol,
                ode_default_options()..., callback=callbacks)
    summary_callback() # print the timer summary

end


### Function to compute DG-USBP operators ###

function compute_DG_UpwindOperators( N, σ )

    # Note: The points, weights, and Vandermonde matrices below
    #  were computed in a separate Matlab script
    if N == 3
        # Gauss-Lobatto nodes
        nodes = [-1.0, 0.0, 1.0]
        # Gauss-Lobatto weights
        weights = [1/3, 4/3, 1/3]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = ( 1/sqrt(6) ) * [
            sqrt(2) -sqrt(3) 1
            sqrt(2) 0 -2
            sqrt(2) sqrt(3) 1
        ]

    elseif N == 4
        # Gauss-Lobatto nodes
        nodes = [-1, -1/sqrt(5), 1/sqrt(5), 1]
        # Gauss-Lobatto weights
        weights = [1/6, 5/6, 5/6, 1/6]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.500000000000000000 -0.645497224367902800 0.499999999999999889 -0.288675134594812921
            0.500000000000000000 -0.288675134594812866 -0.500000000000000000 0.645497224367902800
            0.500000000000000000 0.288675134594812866 -0.500000000000000000 -0.645497224367902800
            0.500000000000000000 0.645497224367902800 0.499999999999999889 0.288675134594812921
        ]

    elseif N == 5
        # Gauss-Lobatto nodes
        nodes = [-1, -sqrt(3/7), 0, sqrt(3/7), 1]
        # Gauss-Lobatto weights
        weights = [1/10, 49/90, 32/45, 49/90, 1/10]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.447213595499957928 -0.591607978309961591 0.500000000000000000 -0.387298334620741758 0.223606797749978908
            0.447213595499957928 -0.387298334620741702 -0.166666666666666685 0.591607978309961591 -0.521749194749950962
            0.447213595499957928 0.000000000000000000 -0.666666666666666741 0.000000000000000000 0.596284793999943941
            0.447213595499957928 0.387298334620741702 -0.166666666666666685 -0.591607978309961591 -0.521749194749950962
            0.447213595499957928 0.591607978309961591 0.500000000000000000 0.387298334620741758 0.223606797749978908
        ]

    elseif N == 6
        # Gauss-Lobatto nodes
        nodes = [
            -1, -sqrt( ( 7+2*sqrt(7) ) / 21 ), -sqrt( ( 7-2*sqrt(7) ) / 21 ),
            sqrt( ( 7-2*sqrt(7) ) / 21 ), sqrt( ( 7+2*sqrt(7) ) / 21 ), 1
        ]
        # Gauss-Lobatto weights
        weights = [
            1/15, (14-sqrt(7))/30, (14+sqrt(7))/30,
            (14+sqrt(7))/30, (14-sqrt(7))/30, 1/15
        ]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [
            0.408248290463863073 -0.547722557505166074 0.483045891539647831 -0.408248290463863017 0.316227766016837830 -0.182574185835055497
            0.408248290463863073 -0.419038058655589740 0.032338332982759031 0.367654222400928044 -0.576443896275456780 0.435014342463467985
            0.408248290463863073 -0.156227735687855862 -0.515384224522407175 0.445155822251155409 0.260216130258618783 -0.526715472069829382
            0.408248290463863073 0.156227735687855862 -0.515384224522407175 -0.445155822251155409 0.260216130258618783 0.526715472069829382
            0.408248290463863073 0.419038058655589740 0.032338332982759031 -0.367654222400928044 -0.576443896275456780 -0.435014342463467985
            0.408248290463863073 0.547722557505166074 0.483045891539647831 0.408248290463863017 0.316227766016837830 0.182574185835055497
        ]

    end

    # Dissipation matrix
    σ_vector = zeros(N)
    σ_vector[N] = σ
    S = U*diagm(σ_vector)*transpose(U);
    # Inverse norm matrix
    P_inv = diagm( 1 ./ weights )
    # Central and upwind SBP operators
    Dc = Matrix(legendre_derivative_operator(-1.0, 1.0, N))
    Dm = Dc - P_inv*S
    Dp = Dc + P_inv*S
    xmin = -1.0
    xmax = +1.0
    D_upw = UpwindOperators(
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dm, 0, nothing),
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dc, 0, nothing),
        MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dp, 0, nothing)
    )

    return D_upw

end
