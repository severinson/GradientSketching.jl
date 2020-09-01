module GradientSketching

using LinearAlgebra, Random

export BiasSEGA, SEGA
export project!, projecta!, gradient, gradient!

"""

Biased SEGA gradient estimator

"""
struct BiasSEGA{T,N}
    h::Array{T,N}
end

BiasSEGA{T}(dims::Tuple) where T = BiasSEGA{T,length(dims)}(zeros(T, dims))
BiasSEGA(dims::Tuple) = BiasSEGA{Float64}(dims)
BiasSEGA{T}(dim::Integer) where T = BiasSEGA{T}((dim,))
BiasSEGA(dim::Integer) = BiasSEGA{Float64}((dim,))

Base.eltype(sega::BiasSEGA{T}) where T = T
Base.size(sega::BiasSEGA) = size(sega.h)
Base.size(sega::BiasSEGA, i::Integer) = size(sega.h, i)
Base.show(io::IO, sega::BiasSEGA) = print(io, "BiasSEGA{$(eltype(sega)),$(size(sega))")

"""
    project!(sega::BiasSEGA{T,1}, s∇::Number, s::AbstractVector; Binv=I) where T

Update the gradient estimate `h` in-place with the one-dimensional gradient sketch `s'*∇`, where
`s` is the sketching vector and `∇` is the gradient. The updated estimate is the projection of
`h` onto the set of arrays `y` satisfying `s'*y = s'*∇`.
"""
function project!(sega::BiasSEGA{T,1}, s∇::Number, s::AbstractVector; Binv=I) where T
    length(s) == size(sega, 1) || throw(DimensionMismatch("s dimensions ($(length(s)),1), h has dimensions $(size(sega))"))
    Binv == I || size(Binv) == (length(s), length(s)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), s has dimensions ($(length(s)),1)"))    
    P = s ./ dot(s, Binv, s)
    sega.h .-= Binv * P * (dot(s, sega.h) - s∇)
end

"""
    project!(sega::BiasSEGA, S∇, S::AbstractMatrix; Binv=I)

Update the gradient estimate `h` in-place with the gradient sketch `S'*∇`, where
`S` is the sketching matrix and `∇` is the gradient. The updated estimate is the projection of
`h` onto the set of arrays `y` satisfying `S'*y = S'*∇`.
"""
function project!(sega::BiasSEGA, S∇, S::AbstractMatrix; Binv=I)
    size(S, 1) == size(sega, 1) || throw(DimensionMismatch("S has dimensions $(size(S)), h has dimensions $(size(sega))"))
    size(S∇)[2:end] == size(sega)[2:end] || throw(DimensionMismatch("S∇ has dimensions $(size(S∇)), h has dimensions $(size(sega))"))
    Binv == I || size(Binv) == (length(s), length(s)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), S has dimensions $(size(S))"))
    StS = S'*Binv*S
    sega.h .-= Binv*S*(StS \ (S'*sega.h .- S∇))
end

project!(sega::BiasSEGA, S∇, s::AbstractVector; Binv=I) = project!(sega, S∇, reshape(s, length(s), 1), Binv=Binv)

"""
    projecta!(sega::BiasSEGA, S∇, S::AbstractMatrix; Binv=I, γ::Integer=5)

Approximate version of `project!` that projects onto each column of `S` separately. Iterates over a random permutation
of the columns `γ` times. This method is equivalent to `project!` if the columns of `S` are orthogonal.
"""
function projecta!(sega::BiasSEGA, S∇, S::AbstractMatrix; Binv=I, γ::Integer=5)
    γ >= 1 || throw(DomainError("γ must be positive."))
    p = collect(1:size(S, 1))
    for _ in 1:γ
        shuffle!(p)
        for i in p
            project!(sega, selectdim(S∇, 1, i:i), view(S, :, i), Binv=Binv)
        end
    end
    sega.h
end

"""
    projecta!(sega::BiasSEGA, S∇::AbstractVector, S::AbstractMatrix; Binv=I, γ::Integer=5)

Approximate version of `project!` for vector gradients.
"""
function projecta!(sega::BiasSEGA, S∇::AbstractVector, S::AbstractMatrix; Binv=I, γ::Integer=5)
    γ >= 1 || throw(DomainError("γ must be positive."))
    p = collect(1:size(S, 1))
    for _ in 1:γ
        shuffle!(p)
        for i in p
            project!(sega, S∇[i], view(S, :, i), Binv=Binv)
        end
    end
    sega.h
end

projecta!(sega::BiasSEGA, S∇, s::AbstractVector; Binv=I, γ::Integer=5) = project!(sega::BiasSEGA, S∇, s::AbstractVector; Binv=Binv)

gradient!(∇, sega::BiasSEGA) = ∇ .= sega.h
gradient(sega::BiasSEGA) = gradient!(zero(sega.h), sega)

"""

SEGA gradient estimator

"""
struct SEGA{T,N}
    θ::T                # Bias removal coefficient
    hp::Array{T,N}      # Previous estimate
    h::BiasSEGA{T,N}    # Biased gradient estimate
    g::Array{T,N}       # Unbiased gradient estimate
end

SEGA{T}(θ, dims::Tuple) where T = SEGA{T,length(dims)}(θ, zeros(T, dims), BiasSEGA{T}(dims), zeros(T, dims))
SEGA(θ, dims::Tuple) = SEGA{Float64}(θ, dims)
SEGA{T}(θ, dim::Integer) where T = SEGA{T}(θ, (dim,))
SEGA(θ, dim::Integer) = SEGA{Float64}(θ, (dim,))

Base.eltype(sega::SEGA{T}) where T = T
Base.size(sega::SEGA) = size(sega.h)
Base.size(sega::SEGA, i::Integer) = size(sega.h, i)
Base.show(io::IO, sega::SEGA) = print(io, "SEGA{$(eltype(sega)),$(size(sega))")

function project!(sega::SEGA, args...; kwargs...)
    gradient!(sega.hp, sega.h)
    project!(sega.h, args...; kwargs...)    
end

function projecta!(sega::SEGA, args...; kwargs...)
    gradient!(sega.hp, sega.h)
    projecta!(sega.h, args...; kwargs...)
end

"""

Compute an unbiased estimate of the gradient.

"""
function gradient!(∇::Array{T,N}, sega::SEGA{T,N}) where {T,N}
    gradient!(∇, sega.h) # store the current (biased) estimate in ∇
    ∇ .*= sega.θ
    ∇ .+= (1-sega.θ) .* sega.hp
end
gradient(sega::SEGA) = gradient!(zero(sega.hp), sega)

end