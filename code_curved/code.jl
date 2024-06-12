
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
using PrettyTables: PrettyTables, pretty_table, ft_printf

const figdir = joinpath(dirname(@__DIR__), "figures")


################################################################################
# Free-stream preservation tests on curvilinear meshes

function free_stream_preservation_generate_data()
    for flux_splitting in [splitting_lax_friedrichs, splitting_vanleer_haenel]
        for polydeg in 3:10
            @info "Generate FSP data for linear boundaries with" polydeg flux_splitting
            t, error_density = compute_fsp_error_degOne(; polydeg, flux_splitting)
            open(joinpath(figdir, "fsp_error_degOne_$(polydeg)_"*string(flux_splitting)*".dat"), "w") do io
                println(io, "# t\tL2_error_density")
                writedlm(io, hcat(t, error_density))
            end

            @info "Generate FSP data for quadratic boundaries with" polydeg flux_splitting
            t, error_density = compute_fsp_error_degTwo(; polydeg, flux_splitting)
            open(joinpath(figdir, "fsp_error_degTwo_$(polydeg)_"*string(flux_splitting)*".dat"), "w") do io
                println(io, "# t\tL2_error_density")
                writedlm(io, hcat(t, error_density))
            end

            @info "Generate FSP data for cubic boundaries with" polydeg flux_splitting
            t, error_density = compute_fsp_error_degThree(; polydeg, flux_splitting)
            open(joinpath(figdir, "fsp_error_degThree_$(polydeg)_"*string(flux_splitting)*".dat"), "w") do io
                println(io, "# t\tL2_error_density")
                writedlm(io, hcat(t, error_density))
            end

            @info "Generate FSP data for quartic boundaries with" polydeg flux_splitting
            t, error_density = compute_fsp_error_degFour(; polydeg, flux_splitting)
            open(joinpath(figdir, "fsp_error_degFour_$(polydeg)_"*string(flux_splitting)*".dat"), "w") do io
                println(io, "# t\tL2_error_density")
                writedlm(io, hcat(t, error_density))
            end
        end
    end
end

# Note the default tolerances for the FSP testing are taken to be quite strict
function compute_fsp_error_degOne(; polydeg = 4, sigma=-0.01,
                                  flux_splitting = splitting_vanleer_haenel,
                                  tspan = (0.0, 10.0),
                                  tol = 1.0e-15)
    equations = CompressibleEulerEquations2D(1.4)

    initial_condition = initial_condition_constant

    # Boundary conditions for free-stream preservation test
    boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

    boundary_conditions = Dict(:Blob1 => boundary_condition_free_stream,
                               :Blob2 => boundary_condition_free_stream)

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( polydeg, sigma ) # construct USBP operator

    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    # unstructured mesh with bi-linear elements
    mesh_file = joinpath(@__DIR__, "mesh_amoeba_deg1.mesh")

    mesh = UnstructuredMesh2D(mesh_file)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions=boundary_conditions)

    ode = semidiscretize(semi, tspan)

    saveat = tspan[end]
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

    sol = solve(ode, SSPRK43();
                abstol = tol, reltol = tol, maxiters = 1e10,
                ode_default_options()..., callback = saving, tstops = saveat)
    return (; t = saved_values.t, error_density = saved_values.saveval)
end

# Note the default tolerances for the FSP testing are taken to be quite strict
function compute_fsp_error_degTwo(; polydeg = 4, sigma=-0.01,
                                  flux_splitting = splitting_vanleer_haenel,
                                  tspan = (0.0, 10.0),
                                  tol = 1.0e-15)
    equations = CompressibleEulerEquations2D(1.4)

    initial_condition = initial_condition_constant

    # Boundary conditions for free-stream preservation test
    boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

    boundary_conditions = Dict(:Blob1 => boundary_condition_free_stream,
                               :Blob2 => boundary_condition_free_stream)

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( polydeg, sigma ) # construct USBP operator

    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    # unstructured mesh with quadratic elements
    mesh_file = joinpath(@__DIR__, "mesh_amoeba_deg2.mesh")

    mesh = UnstructuredMesh2D(mesh_file)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions=boundary_conditions)

    ode = semidiscretize(semi, tspan)

    saveat = tspan[end]
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

    sol = solve(ode, SSPRK43();
                abstol = tol, reltol = tol, maxiters = 1e10,
                ode_default_options()..., callback = saving, tstops = saveat)
    return (; t = saved_values.t, error_density = saved_values.saveval)
end

# Note the default tolerances for the FSP testing are taken to be quite strict
function compute_fsp_error_degThree(; polydeg = 4, sigma=-0.01,
                                    flux_splitting = splitting_vanleer_haenel,
                                    tspan = (0.0, 10.0),
                                    tol = 1.0e-15)
    equations = CompressibleEulerEquations2D(1.4)

    initial_condition = initial_condition_constant

    # Boundary conditions for free-stream preservation test
    boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

    boundary_conditions = Dict(:Blob1 => boundary_condition_free_stream,
                               :Blob2 => boundary_condition_free_stream)

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( polydeg, sigma ) # construct USBP operator

    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    # unstructured mesh with quadratic elements
    mesh_file = joinpath(@__DIR__, "mesh_amoeba_deg3.mesh")

    mesh = UnstructuredMesh2D(mesh_file)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions=boundary_conditions)

    ode = semidiscretize(semi, tspan)

    saveat = tspan[end]
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

    sol = solve(ode, SSPRK43();
                abstol = tol, reltol = tol, maxiters = 1e10,
                ode_default_options()..., callback = saving, tstops = saveat)
    return (; t = saved_values.t, error_density = saved_values.saveval)
end

# Note the default tolerances for the FSP testing are taken to be quite strict
function compute_fsp_error_degFour(; polydeg = 4, sigma=-0.01,
                                    flux_splitting = splitting_vanleer_haenel,
                                    tspan = (0.0, 10.0),
                                    tol = 1.0e-15)
    equations = CompressibleEulerEquations2D(1.4)

    initial_condition = initial_condition_constant

    # Boundary conditions for free-stream preservation test
    boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

    boundary_conditions = Dict(:Blob1 => boundary_condition_free_stream,
                               :Blob2 => boundary_condition_free_stream)

    # Compute the DG-USBP operators
    D_upw = compute_DG_UpwindOperators( polydeg, sigma ) # construct USBP operator

    solver = FDSBP(D_upw,
                   surface_integral = SurfaceIntegralStrongForm(FluxUpwind(flux_splitting)),
                   volume_integral = VolumeIntegralUpwind(flux_splitting))

    # unstructured mesh with quadratic elements
    mesh_file = joinpath(@__DIR__, "mesh_amoeba_deg4.mesh")

    mesh = UnstructuredMesh2D(mesh_file)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions=boundary_conditions)

    ode = semidiscretize(semi, tspan)

    saveat = tspan[end]
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

    sol = solve(ode, SSPRK43();
                abstol = tol, reltol = tol, maxiters = 1e10,
                ode_default_options()..., callback = saving, tstops = saveat)
    return (; t = saved_values.t, error_density = saved_values.saveval)
end


### Function to compute DG-USBP operators ###

function compute_DG_UpwindOperators(N, σ)

    # Note: The points, weights, and Vandermonde matrices below
    #  were computed in a separate Matlab script
    if N == 3
        # Accurate up to 1st order polynomials (analogous to 1st order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [-1.0, 0.0, 1.0]
        # Gauss-Lobatto weights
        weights = [1 / 3, 4 / 3, 1 / 3]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = (1 / sqrt(6)) * [sqrt(2) -sqrt(3) 1
                             sqrt(2) 0 -2
                             sqrt(2) sqrt(3) 1]
    elseif N == 4
        # Accurate up to 2nd order polynomials (analogous to 2nd order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [-1, -1 / sqrt(5), 1 / sqrt(5), 1]
        # Gauss-Lobatto weights
        weights = [1 / 6, 5 / 6, 5 / 6, 1 / 6]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.500000000000000000 -0.645497224367902800 0.499999999999999889 -0.288675134594812921
             0.500000000000000000 -0.288675134594812866 -0.500000000000000000 0.645497224367902800
             0.500000000000000000 0.288675134594812866 -0.500000000000000000 -0.645497224367902800
             0.500000000000000000 0.645497224367902800 0.499999999999999889 0.288675134594812921]
    elseif N == 5
        # Accurate up to 3rd order polynomials (analogous to 3rd order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [-1, -sqrt(3 / 7), 0, sqrt(3 / 7), 1]
        # Gauss-Lobatto weights
        weights = [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.447213595499957928 -0.591607978309961591 0.500000000000000000 -0.387298334620741758 0.223606797749978908
             0.447213595499957928 -0.387298334620741702 -0.166666666666666685 0.591607978309961591 -0.521749194749950962
             0.447213595499957928 0.000000000000000000 -0.666666666666666741 0.000000000000000000 0.596284793999943941
             0.447213595499957928 0.387298334620741702 -0.166666666666666685 -0.591607978309961591 -0.521749194749950962
             0.447213595499957928 0.591607978309961591 0.500000000000000000 0.387298334620741758 0.223606797749978908]
    elseif N == 6
        # Accurate up to 4th order polynomials (analogous to 4th order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [
            -1, -sqrt((7 + 2 * sqrt(7)) / 21), -sqrt((7 - 2 * sqrt(7)) / 21),
            sqrt((7 - 2 * sqrt(7)) / 21), sqrt((7 + 2 * sqrt(7)) / 21), 1,
        ]
        # Gauss-Lobatto weights
        weights = [
            1 / 15, (14 - sqrt(7)) / 30, (14 + sqrt(7)) / 30,
            (14 + sqrt(7)) / 30, (14 - sqrt(7)) / 30, 1 / 15,
        ]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.408248290463863073 -0.547722557505166074 0.483045891539647831 -0.408248290463863017 0.316227766016837830 -0.182574185835055497
             0.408248290463863073 -0.419038058655589740 0.032338332982759031 0.367654222400928044 -0.576443896275456780 0.435014342463467985
             0.408248290463863073 -0.156227735687855862 -0.515384224522407175 0.445155822251155409 0.260216130258618783 -0.526715472069829382
             0.408248290463863073 0.156227735687855862 -0.515384224522407175 -0.445155822251155409 0.260216130258618783 0.526715472069829382
             0.408248290463863073 0.419038058655589740 0.032338332982759031 -0.367654222400928044 -0.576443896275456780 -0.435014342463467985
             0.408248290463863073 0.547722557505166074 0.483045891539647831 0.408248290463863017 0.316227766016837830 0.182574185835055497]
    elseif N == 7
        # Accurate up to 5th order polynomials (analogous to 5th order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [-1.000000000000000000
                 -0.830223896278566964
                 -0.468848793470714231
                  0.000000000000000000
                  0.468848793470714231
                  0.830223896278566964
                  1.000000000000000000]
        # Gauss-Lobatto weights
        weights = [0.047619047619047616
                   0.276826047361565908
                   0.431745381209862611
                   0.487619047619047619
                   0.431745381209862611
                   0.276826047361565908
                   0.047619047619047616]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.377964473009227198 -0.511766315719158982 0.462910049886275765 -0.408248290463862962 0.345032779671177126 -0.267261241912424397 0.154303349962091968
            0.377964473009227198 -0.424880624620487368 0.146463711889623649 0.182383954787603830 -0.444160122660271606 0.534988362357666447 -0.372039008278410188
            0.377964473009227198 -0.239941019663880289 -0.331627731844133855 0.547785931153189187 -0.176898880747847265 -0.377304758402661478 0.464621018255665208
            0.377964473009227198 0.000000000000000000 -0.555492059863530785 -0.000000000000000025 0.552052447473883379 0.000000000000000034 -0.493770719878694142
            0.377964473009227198 0.239941019663880289 -0.331627731844133855 -0.547785931153189187 -0.176898880747847320 0.377304758402661422 0.464621018255665319
            0.377964473009227198 0.424880624620487368 0.146463711889623677 -0.182383954787603830 -0.444160122660271606 -0.534988362357666447 -0.372039008278410188
            0.377964473009227198 0.511766315719158982 0.462910049886275765 0.408248290463862962 0.345032779671177126 0.267261241912424397 0.154303349962091968]
    elseif N == 8
        # Accurate up to 6th order polynomials (analogous to 6th order boundary closure)
        # Gauss-Lobatto nodes
        nodes = [-1.000000000000000000
                 -0.871740148509606572
                 -0.591700181433142292
                 -0.209299217902478851
                 0.209299217902478851
                 0.591700181433142292
                 0.871740148509606572
                 1.000000000000000000]
        # Gauss-Lobatto weights
        weights = [0.035714285714285712
                   0.210704227143506145
                   0.341122692483504408
                   0.412458794658703720
                   0.412458794658703720
                   0.341122692483504408
                   0.210704227143506145
                   0.035714285714285712]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.353553390593273731 -0.481812055829715757 0.443202630213959226 -0.400891862868636650 0.353553390593273786 -0.298807152333598280 0.231455024943137772 -0.133630620956212198
             0.353553390593273731 -0.420014913102715270 0.212670894741185279 0.050439319495041914 -0.299316079318915684 0.464425703690528491 -0.490082620780553924 0.324579903216069499
             0.353553390593273731 -0.285088280851118092 -0.180869637101319231 0.497617644822938299 -0.403907085458223847 -0.023238017137308296 0.423255526474823673 -0.412990733844903868
             0.353553390593273731 -0.100842886461144990 -0.475003887853825135 0.298526831114831914 0.349669774183865745 -0.441002319750553318 -0.164627930637407521 0.454124869754291172
             0.353553390593273731 0.100842886461144990 -0.475003887853825135 -0.298526831114831914 0.349669774183865745 0.441002319750553318 -0.164627930637407577 -0.454124869754291172
             0.353553390593273731 0.285088280851118092 -0.180869637101319231 -0.497617644822938299 -0.403907085458223847 0.023238017137308400 0.423255526474823673 0.412990733844903812
             0.353553390593273731 0.420014913102715270 0.212670894741185279 -0.050439319495041914 -0.299316079318915684 -0.464425703690528491 -0.490082620780553924 -0.324579903216069499
             0.353553390593273731 0.481812055829715757 0.443202630213959226 0.400891862868636650 0.353553390593273786 0.298807152333598280 0.231455024943137716 0.133630620956212198]
    elseif N == 9
        # Accurate up to 7th order polynomials (analogous to 7th order bounadry closure)
        # Gauss-Lobatto nodes
        nodes = [-1.000000000000000000
                 -0.899757995411460176
                 -0.677186279510737732
                 -0.363117463826178155
                  0.000000000000000000
                  0.363117463826178155
                  0.677186279510737732
                  0.899757995411460176
                  1.000000000000000000]
        # Gauss-Lobatto weights
        weights = [0.027777777777777776
                   0.165495361560805576
                   0.274538712500161652
                   0.346428510973046166
                   0.371519274376417241
                   0.346428510973046166
                   0.274538712500161652
                   0.165495361560805576
                   0.027777777777777776]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.333333333333333315 -0.456435464587638451 0.424918292799398800 -0.390867979985285718 0.353553390593273786 -0.311804782231161703 0.263523138347364960 -0.204124145231931592 0.117851130197757906
             0.333333333333333315 -0.410681458652072118 0.251519259948122964 -0.040737597829788391 -0.178191511157030885 0.358765167661249884 -0.459485642142611950 0.448295373738632374 -0.287658966104661462
             0.333333333333333315 -0.309091834100857998 -0.068064695553868543 0.400669816937080814 -0.471976206965718015 0.234688035512350623 0.155644434260338665 -0.434565831462998931 0.370498793857794995
             0.333333333333333315 -0.165739688301386606 -0.365562404165425181 0.430146914601147345 0.054177716836944619 -0.467960316948057897 0.281253510309641908 0.261757768447749506 -0.416190167455362969
             0.333333333333333315 0.000000000000000017 -0.485620906056455692 0.000000000000000027 0.484873221385061282 -0.000000000000000035 -0.481870881549467334 -0.000000000000000006 0.430998419008943090
             0.333333333333333315 0.165739688301386662 -0.365562404165425181 -0.430146914601147401 0.054177716836944639 0.467960316948057897 0.281253510309641908 -0.261757768447749450 -0.416190167455363025
             0.333333333333333315 0.309091834100857998 -0.068064695553868529 -0.400669816937080814 -0.471976206965718015 -0.234688035512350623 0.155644434260338665 0.434565831462998931 0.370498793857795106
             0.333333333333333315 0.410681458652072118 0.251519259948122964 0.040737597829788350 -0.178191511157030885 -0.358765167661249940 -0.459485642142611950 -0.448295373738632319 -0.287658966104661462
             0.333333333333333315 0.456435464587638451 0.424918292799398800 0.390867979985285718 0.353553390593273786 0.311804782231161703 0.263523138347364960 0.204124145231931592 0.117851130197757892]
    elseif N == 10
        # Accurate up to 8th order polynomials (analogous to 8th order bounadry closure)
        # Gauss-Lobatto nodes
        nodes = [
            -1.000000000000000000,
            -0.919533908166458858,
            -0.738773865105505134,
            -0.477924949810444477,
            -0.165278957666387005,
            0.165278957666387005,
            0.477924949810444477,
            0.738773865105505134,
            0.919533908166458858,
            1.000000000000000000,
        ]
        # Gauss-Lobatto weights
        weights = [
            0.022222222222222223,
            0.133305990851070061,
            0.224889342063126441,
            0.292042683679683779,
            0.327539761183897438,
            0.327539761183897438,
            0.292042683679683779,
            0.224889342063126441,
            0.133305990851070061,
            0.022222222222222223,
        ]
        # Vandermonde matrix w.r.t. discrete orthonormal polynomials
        U = [0.316227766016837941 -0.434613493680176599 0.408248290463862962 -0.380058475033045906 0.349602949390050433 -0.316227766016837886 0.278886675511358595 -0.235702260395515895 0.182574185835055580 -0.105409255338946073
             0.316227766016837941 -0.399641844385611344 0.274252237450055780 -0.103678272577398103 -0.083990014960314149 0.257207720890767100 -0.385561337322198383 0.443540759006515295 -0.411185898108048287 0.258172414145150575
             0.316227766016837941 -0.321081090553111070 0.014205815965851963 0.299952149603675000 -0.455844726466966388 0.370665888462330340 -0.088833105264021300 -0.238951395342708733 0.429083278110615973 -0.335327706924976932
             0.316227766016837941 -0.207712632154040316 -0.261125385144171107 0.456547050005304655 -0.173989077148133220 -0.290370313388308432 0.448867621021733887 -0.134476737163079146 -0.316321344624004963 0.382127389544169971
             0.316227766016837941 -0.071832465223206465 -0.435580958735599766 0.215406291529543686 0.364220869185363338 -0.334867240182410264 -0.253359853946872604 0.415372456200592710 0.115849778786381835 -0.404684915201874995
             0.316227766016837941 0.071832465223206465 -0.435580958735599766 -0.215406291529543686 0.364220869185363338 0.334867240182410264 -0.253359853946872604 -0.415372456200592710 0.115849778786381835 0.404684915201874995
             0.316227766016837941 0.207712632154040316 -0.261125385144171107 -0.456547050005304655 -0.173989077148133220 0.290370313388308432 0.448867621021733887 0.134476737163079146 -0.316321344624004908 -0.382127389544169971
             0.316227766016837941 0.321081090553111070 0.014205815965851963 -0.299952149603675000 -0.455844726466966388 -0.370665888462330340 -0.088833105264021300 0.238951395342708733 0.429083278110615973 0.335327706924976932
             0.316227766016837941 0.399641844385611344 0.274252237450055780 0.103678272577398062 -0.083990014960314149 -0.257207720890767100 -0.385561337322198383 -0.443540759006515295 -0.411185898108048287 -0.258172414145150575
             0.316227766016837941 0.434613493680176599 0.408248290463862962 0.380058475033045906 0.349602949390050433 0.316227766016837886 0.278886675511358595 0.235702260395515895 0.182574185835055580 0.105409255338946073]
    end

    # Dissipation matrix
    σ_vector = zeros(N)
    σ_vector[N] = σ
    S = U * diagm(σ_vector) * transpose(U)
    # Inverse norm matrix
    P_inv = diagm(1 ./ weights)
    # Central and upwind SBP operators
    Dc = Matrix(legendre_derivative_operator(-1.0, 1.0, N))
    Dm = Dc - P_inv * S
    Dp = Dc + P_inv * S
    xmin = -1.0
    xmax = +1.0
    D_upw = UpwindOperators(MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dm, 0,
                                                     nothing),
                            MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dc, 0,
                                                     nothing),
                            MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dp, 0,
                                                     nothing))

    return D_upw
end
