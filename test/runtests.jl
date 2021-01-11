using GradientSketching
using Test, Random, LinearAlgebra

@time @testset "Projection" begin
    # vector gradient with vector sketches
    h = zeros(2)
    project!(h, 1, [1, 0])
    project!(h, 2, [0, 1])
    @test h ≈ [1, 2]

    y = randn(2)
    Q = qr(randn(2, 2)).Q # orthonormal matrix
    Qy = Q'*y
    project!(h, Qy[1], Q[:, 1])
    project!(h, Qy[2], Q[:, 2])
    @test h ≈ y

    # vector gradient with coordinate sketches
    h = zeros(2)
    project!(h, 10, 1)
    @test h[1] == 10
    @test h[2] == 0
    project!(h, 20, 2)
    @test h[1] == 10
    @test h[2] == 20

    # matrix gradient with vector sketches
    h = zeros(2, 2)
    project!(h, [1, 2], [1, 0])
    project!(h, [3, 4], [0, 1])
    @test h ≈ [1 2; 3 4]

    Y = randn(2, 2)
    QY = Q'*Y
    project!(h, QY[1, :], Q[:, 1])
    project!(h, QY[2, :], Q[:, 2])
    @test h ≈ Y

    # vector gradient with matrix sketches
    h = zeros(2)
    project!(h, [1, 2], [1 0;0 1])
    @test h ≈ [1, 2]

    S = randn(2, 2)
    Sy = S'*y
    project!(h, Sy, S)
    @test h ≈ y

    # matrix gradient with matrix sketches
    h = zeros(2, 2)
    project!(h, [1 2; 3 4], [1 0;0 1])
    @test h ≈ [1 2; 3 4]

    SY = S'*Y
    project!(h, SY, S)
    @test h ≈ Y

    # vector of arrays gradient with vector sketches
    h = [zeros(2, 2), zeros(2, 2)]
    project!(h, ones(2, 2), [1, 0])
    project!(h, 2.0.*ones(2, 2), [0, 1])
    @test h[1] ≈ ones(2, 2)
    @test h[2] ≈ 2.0.*ones(2, 2)

    # same as above but saving to a view
    for hi in h
        hi .= 0
    end
    project!(view(h, :), ones(2, 2), [1, 0])
    project!(view(h, :), 2.0.*ones(2, 2), [0, 1])    
    @test h[1] ≈ ones(2, 2)
    @test h[2] ≈ 2.0.*ones(2, 2)

    # vector of arrays gradient with coordinate sketches
    h = [zeros(2, 2), zeros(2, 2)]
    v1 = fill(10, 2, 2)
    v2 = fill(20, 2, 2)
    project!(h, v1, 1)
    @test h[1] ≈ v1
    @test h[2] ≈ zeros(2, 2)
    project!(h, v2, 2)
    @test h[1] ≈ v1
    @test h[2] ≈ v2

    # vector of arrays gradient with matrix sketches
    h = [zeros(2, 2), zeros(2, 2)]
    project!(h, [ones(2, 2), 2.0.*ones(2, 2)], [1 0;0 1])
    @test h[1] ≈ ones(2, 2)
    @test h[2] ≈ 2.0.*ones(2, 2)

    S = randn(2, 2)
    S∇ = [
        S[1, 1] .* ones(2, 2) .+ S[2, 1] .* ones(2, 2),
        S[1, 2] .* ones(2, 2) .+ S[2, 2] .* ones(2, 2),
    ]
    project!(h, S∇, S)
    @test h[1] ≈ ones(2, 2)
    @test h[2] ≈ ones(2, 2)

    # same as the previous, but saving the result to a view
    for hi in h
        hi .= 0
    end
    project!(view(h, :), S∇, S)
    @test h[1] ≈ ones(2, 2)
    @test h[2] ≈ ones(2, 2)

    # try for many random matrices to ensure there are no problem with singular StS matrices
    h = ones(Float64, 5)
    S = zeros(Float64, 5, 10)
    for _ in 1:100
        S .= randn(5, 10)
        h .= 1
        project!(h, S'*h, S)
        @test h ≈ ones(Float64, 5)
    end
end

@time @testset "BiasSEGA" begin

    ### Constructors
    sega = BiasSEGA((10, 5))
    @test eltype(sega) == Float64
    @test size(sega) == (10, 5)    

    sega = BiasSEGA{Float32}((10, 5))
    @test eltype(sega) == Float32
    @test size(sega) == (10, 5)        

    sega = BiasSEGA(3)
    @test eltype(sega) == Float64
    @test size(sega) == (3, )          

    sega = BiasSEGA{Float32}(3)
    @test eltype(sega) == Float32
    @test size(sega) == (3, )

    sega = BiasSEGA(zeros(5))
    @test eltype(sega) == Float64
    @test size(sega) == (5, )

    ### Vector sketches
    sega = BiasSEGA(2)
    ∇ = zeros(2)

    # Project onto the first dimension
    project!(sega, 1, [1, 0])
    @test gradient(sega) ≈ [1, 0]

    # Project onto the second dimension
    project!(sega, 1, [0, 1])
    @test gradient(sega) ≈ [1, 1]

    ### 1-dimensional matrix sketches for 2-dimensional gradients
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)

    # Project onto the first dimension
    correct[1, :] .= 1
    project!(sega, ones(1, 3), [1 0]')
    @test gradient(sega) ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    project!(sega, ones(1, 3), [0 1]')
    @test gradient(sega) ≈ correct

    # Vector sketches for 2-dimensional gradients
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)

    # Project onto the first dimension
    correct[1, :] .= 1
    project!(sega, ones(3), [1, 0])
    @test gradient(sega) ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    project!(sega, ones(3), [0, 1])
    @test gradient(sega) ≈ correct  

    ### Matrix sketches
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1.0 2; 3 4]
    project!(sega, S'*correct, S)    
    @test gradient(sega) ≈ correct

    ### Vector sketches with approximate projection
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)
    
    # Project onto the first dimension
    correct[1, :] .= 1
    projecta!(sega, ones(3), [1, 0])
    @test gradient(sega) ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    projecta!(sega, ones(3), [0, 1])
    @test gradient(sega) ≈ correct     

    ### Matrix sketches for 1-dimensional gradient with approximate projection
    # Test that we get arbitrarily close after a very large number of iterations
    sega = BiasSEGA(2)
    ∇ = zeros(2)
    correct = ones(2)
    S = [1 2; 3 4]
    projecta!(sega, S'*correct, S, γ=10000)    
    @test gradient(sega) ≈ correct 

    ### Matrix sketches for 2-dimensional gradient with approximate projection
    # Test that we get arbitrarily close after a very large number of iterations
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1 2; 3 4]
    projecta!(sega, S'*correct, S, γ=10000)    
    @test gradient(sega) ≈ correct
end

@time @testset "SEGA" begin

    ### Constructors
    sega = SEGA((10, 5))
    @test eltype(sega) == Float64
    @test size(sega) == (10, 5)    

    sega = SEGA{Float32}((10, 5))
    @test eltype(sega) == Float32
    @test size(sega) == (10, 5)        

    sega = SEGA(3)
    @test eltype(sega) == Float64
    @test size(sega) == (3, )          

    sega = SEGA{Float32}(3)
    @test eltype(sega) == Float32
    @test size(sega) == (3, )      

    sega = SEGA(zeros(5))
    @test eltype(sega) == Float64
    @test size(sega) == (5, )    

    sega = SEGA(zeros(5), zeros(5), zeros(5))
    @test eltype(sega) == Float64
    @test size(sega) == (5, )

    ### Matrix sketches
    sega = SEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1.0 2; 3 4]
    project!(sega, 1, S'*correct, S)
    unbias!(sega)
    @test gradient(sega) ≈ correct

    ### Test SEGA un-biasing
    h = ones(2)
    sega = SEGA(h)
    
    project!(sega, 1/2, 0, 1)
    g = unbias!(sega)
    @test g ≈ [-1, 1]

    project!(sega, 1/2, 0, 2)
    g = unbias!(sega)
    @test g ≈ [0, -1]

    ### Same as previous, but for a vector of arrays gradient
    h = [ones(2) for _ in 1:2]
    sega = SEGA(h)

    project!(sega, 1/2, zeros(2), 1)
    g = unbias!(sega)
    @test g ≈ [fill(-1, 2), ones(2)]

    project!(sega, 1/2, zeros(2), 2)
    g = unbias!(sega)
    @test g ≈ [zeros(2), fill(-1, 2)]

    ### Test SEGA un-biasing when minimizing f(x) = x'*Q*x ./ 2 .+ c'*x
    # over many realizations of the starting point and current gradient estimate
    Random.seed!(123)
    Q = Symmetric(randn(2, 2))
    c = randn(2)
    f = (x) -> x'*Q*x ./ 2 .+ c'*x
    ∇ = (x) -> Q*x .+ c    
    x = randn(2)
    h = zeros(2)
    s = zeros(2)
    avg = zeros(2)
    nsamples = 100000
    for i in 1:nsamples
        x .= randn(2)
        s .= randn(2)
        h .= randn(2)
        sega = SEGA(h)
        project!(sega, 1/2, dot(s, ∇(x)), s)
        g = unbias!(sega)
        avg .+= g .- ∇(x)
    end
    avg ./= nsamples * length(avg)
    @test avg ≈ zeros(2) atol=1e-2
end
