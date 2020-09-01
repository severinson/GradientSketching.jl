using GradientSketching
using Test

@testset "BiasSEGA" begin

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

    ### Vector sketches
    sega = BiasSEGA(2)
    ∇ = zeros(2)

    # Project onto the first dimension
    project!(sega, 1, [1, 0])
    @test gradient(sega) ≈ [1, 0]
    gradient!(∇, sega)
    @test ∇ ≈ [1, 0]    

    # Project onto the second dimension
    project!(sega, 1, [0, 1])
    @test gradient(sega) ≈ [1, 1]
    gradient!(∇, sega)
    @test ∇ ≈ [1, 1]

    ### 1-dimensional matrix sketches for 2-dimensional gradients
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)

    # Project onto the first dimension
    correct[1, :] .= 1
    project!(sega, ones(1, 3), [1 0]')
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    project!(sega, ones(1, 3), [0 1]')
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    # Vector sketches for 2-dimensional gradients
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)

    # Project onto the first dimension
    correct[1, :] .= 1
    project!(sega, ones(1, 3), [1, 0])
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    project!(sega, ones(1, 3), [0, 1])
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct    

    ### Matrix sketches
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1 2; 3 4]
    project!(sega, S'*correct, S)    
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    ### Vector sketches with approximate projection
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = zeros(2, 3)
    
    # Project onto the first dimension
    correct[1, :] .= 1
    projecta!(sega, ones(1, 3), [1, 0])
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    # Project onto the second dimension
    correct[2, :] .= 1
    projecta!(sega, ones(1, 3), [0, 1])
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct        

    ### Matrix sketches for 1-dimensional gradient with approximate projection
    # Test that we get arbitrarily close after a very large number of iterations
    sega = BiasSEGA(2)
    ∇ = zeros(2)
    correct = ones(2)
    S = [1 2; 3 4]
    projecta!(sega, S'*correct, S, γ=10000)    
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct    

    ### Matrix sketches for 2-dimensional gradient with approximate projection
    # Test that we get arbitrarily close after a very large number of iterations
    sega = BiasSEGA((2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1 2; 3 4]
    projecta!(sega, S'*correct, S, γ=10000)    
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct
end

@testset "SEGA" begin

    ### Constructors
    sega = SEGA(1, (10, 5))
    @test eltype(sega) == Float64
    @test size(sega) == (10, 5)    

    sega = SEGA{Float32}(1, (10, 5))
    @test eltype(sega) == Float32
    @test size(sega) == (10, 5)        

    sega = SEGA(1, 3)
    @test eltype(sega) == Float64
    @test size(sega) == (3, )          

    sega = SEGA{Float32}(1, 3)
    @test eltype(sega) == Float32
    @test size(sega) == (3, )      

    ### Matrix sketches
    sega = SEGA(1, (2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1 2; 3 4]
    project!(sega, S'*correct, S)
    @test gradient(sega) ≈ correct
    gradient!(∇, sega)
    @test ∇ ≈ correct

    ### Matrix sketches with θ != 1
    sega = SEGA(1/2, (2, 3))
    ∇ = zeros(2, 3)
    correct = ones(2, 3)
    S = [1 2; 3 4]
    project!(sega, S'*correct, S)
    @test gradient(sega) ≈ correct ./ 2
    gradient!(∇, sega)
    @test ∇ ≈ correct ./ 2
end
