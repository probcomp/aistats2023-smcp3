using CairoMakie
using CairoMakie.Colors

include("ula_hmm.jl")

function get_vals_and_logweights(n, y)
    vals = Float64[]
    weights = Float64[]
    for _=1:n
        (val, logweight) = smcp3_generate_and_weight_step_ula_L(0., y)
        push!(vals, val)
        push!(weights, logweight)
    end
    return (vals, weights)
end

function get_exact_q(proposed_x, y)
    f(aux, p) = (let (m, s) = ula_params(0, aux, y)
        exp(
            logpdf(normal, aux, 0, X_STD()) +
            logpdf(normal, proposed_x, m, s)
        )
    end)
    prob = IntegralProblem(f,-Inf, Inf)
    sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
    return sol.u
end

function P_y(y)
    std = sqrt(X_STD()^2 + Y_STD()^2)
    return exp(logpdf(normal, y, 0, std))
end
function joint_pdf(x, y)
    joint_p = exp(Gen.assess(initial_model, (), choicemap((:xₜ, x), (:yₜ, y)))[1])
    return joint_p
end
@memoize function posterior_curve(xs, y)
    [normalized_pdf(x, y) for x in xs]
end
@memoize function joint_curve(xs, y)
    [joint_pdf(x, y) for x in xs]
end
@memoize function proposal_curve(xs, y)
    [get_exact_q(x, y) for x in xs]
end
function plot_posterior!(ax, y)
    x0 = min(-2, y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, posterior_curve(xs, y), linewidth=4, color=RGBA(0, 0, 0, 0.2))
end
function plot_joint!(ax, y)
    x0 = min(-2, y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, 4 * joint_curve(xs, y), linewidth=4, color=RGBA(0, 0, 0, 0.2))
end
function plot_proposal!(ax, y)
    x0 = min(-2, y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, 1/400 * proposal_curve(xs, y), linewidth=4, color=RGBA(1., 0, 0, 0.2))
end
function plot_joint_over_prop!(ax, y)
    x0 = min(-2, y - 2)
    x1 = max(2, y + 2)
    xs = x0:0.01:x1
    lines!(ax, xs, joint_curve(xs, y) ./ proposal_curve(xs, y) , linewidth=4, color=RGBA(0, 1., 0, 0.2))
end

function get_vals_and_logweights(n, y)
    vals = Float64[]
    weights = Float64[]
    for _=1:n
        (val, logweight) = smcp3_generate_and_weight_initial(y)
        push!(vals, val)
        push!(weights, logweight)
    end
    return (vals, weights)
end

function make_plot(y, n = 1000)
    f = Figure()
    ax = Axis(f[1, 1])

    plot_joint!(ax, y)
    plot_proposal!(ax, y)
    plot_joint_over_prop!(ax, y)
    (vals, logweights) = get_vals_and_logweights(n, y)

    # p_y = exp(logpdf(normal, 5, 0, 2))
    scatter!(ax, vals, exp.(logweights); color=RGBA(0, 0, 1., 0.1))

    ylims!(ax, (-.001, .01))

    return f
end
make_plot(5.0)