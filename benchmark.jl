# Benchmark script for comparing the convergence rate of GD, BiasSEGA, and SEGA
# for minimizing a multivariate quadratic on the unit ball

using Random, LinearAlgebra

function main(n=10, γ=1/(2n), niterations=30)

    Random.seed!(123)
    Q = Symmetric(randn(n, n))
    c = randn(n)
    f = (x) -> x'*Q*x ./ 2 .+ c'*x
    ∇ = (x) -> Q*x .+ c

    xgd = zeros(n, niterations)
    xbias = zeros(n, niterations)
    xsega = zeros(n, niterations)

    bsega = BiasSEGA(n)
    sega = SEGA(n)

    s = zeros(n)
    for i in 2:niterations
        # GD
        xgd[:, i] .= xgd[:, i-1] .- γ.*∇(xgd[:, i-1])
        xgd[:, i] ./= norm(xgd[:, i])

        # BiasSEGA
        s .= randn(n)
        project!(bsega, dot(s, ∇(xbias[:, i-1])), s)
        xbias[:, i] .= xbias[:, i-1] .- γ.*gradient(bsega)
        xbias[:, i] ./= norm(xbias[:, i])        

        # SEGA
        project!(sega, 1/n, dot(s, ∇(xsega[:, i-1])), s)
        unbias!(sega)
        xsega[:, i] .= xsega[:, i-1] .- γ.*gradient(sega)
        xsega[:, i] ./= norm(xsega[:, i])        
    end

    println("GD \t BiasSEGA \t SEGA")
    for i in 2:niterations
        println("$(f(xgd[:, i])) \t $(f(xbias[:, i])) \t $(f(xsega[:, i]))")
    end
end