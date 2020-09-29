module GradientSketching

using LinearAlgebra, Random, CatViews

export BiasSEGA, SEGA
export project!, projecta!, gradient, gradient!

### General projection methods

"""
    project!(h::AbstractVector, s∇::Number, s::AbstractVector; Binv=I)

Project `h` onto the space consisting of all `y` such that `s'*y = s∇`. Updates `h` in-place.

"""
function project!(h::AbstractVector, s∇::Number, s::AbstractVector; Binv=I)
    length(s) == length(h) || throw(DimensionMismatch("s has length $(length(s)), h has length $(length(h))"))
    Binv == I || size(Binv) == (length(s), length(s)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), s has dimensions ($(length(s)),1)"))
    P = Binv * (s ./ dot(s, Binv, s))
    c = dot(s, h) - s∇
    for i in 1:length(h)
        h[i] -= c*P[i]
    end
    h
end

function project!(h::AbstractMatrix, s∇::AbstractVector, s::AbstractVector; Binv=I) where T
    length(s) == size(h, 1) || throw(DimensionMismatch("s has length $(length(s)), h has dimensions $(size(h))"))
    length(s∇) == size(h, 2) || throw(DimensionMismatch("s∇ has length $(length(s∇)), h has dimensions $(size(h))"))
    Binv == I || size(Binv) == (length(s), length(s)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), s has dimensions ($(length(s)),1)"))
    P = Binv * (s ./ dot(s, Binv, s))
    for col in 1:size(h, 2)
        hv = view(h, :, col)
        c  = dot(s, hv) - s∇[col]
        for row in 1:length(hv)
            hv[row] -= c*P[row]
        end
    end
    h
end

function project!(h::AbstractVector, S∇::AbstractVector, S::AbstractMatrix; Binv=I)
    size(S, 1) == size(h, 1) || throw(DimensionMismatch("S has dimensions $(size(S)), h has dimensions $(size(h))"))
    size(S∇, 2) == size(h, 2) || throw(DimensionMismatch("S∇ has dimensions $(size(S∇)), h has dimensions $(size(h))"))
    Binv == I || error("not implemented")
    Binv == I || size(Binv) == (size(S, 1), size(S, 1)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), S has dimensions $(size(S))"))
    h .-= S * (S \ (S' \ (S'*h .- S∇)))
end

function project!(h::AbstractMatrix, S∇::AbstractMatrix, S::AbstractMatrix; Binv=I)
    size(S, 1) == size(h, 1) || throw(DimensionMismatch("S has dimensions $(size(S)), h has dimensions $(size(h))"))
    size(S∇, 2) == size(h, 2) || throw(DimensionMismatch("S∇ has dimensions $(size(S∇)), h has dimensions $(size(h))"))
    Binv == I || error("not implemented")
    Binv == I || size(Binv) == (size(S, 1), size(S, 1)) || throw(DimensionMismatch("Binv has dimensions $(size(Binv)), S has dimensions $(size(S))"))
    h .-= S * (S \ (S' \ (S'*h .- S∇)))
end

function project!(h::AbstractVector{T1}, S∇::AbstractVector{T2}, S::AbstractMatrix; Binv=I) where {T1<:AbstractArray{Tv1,N},T2<:AbstractArray{Tv2,N}} where {Tv1,Tv2,N}
    project!(
        reshape(CatView(h...), length(h[1]), length(h))', # convert h to a matrix by unrolling the component arrays
        reshape(CatView(S∇...), length(S∇[1]), length(S∇))',
        S, Binv=Binv,
    )
    h
end

function project!(h::AbstractVector{T1}, S∇::T2, S::AbstractVector; Binv=I) where {T1<:AbstractArray{Tv1,N},T2<:AbstractArray{Tv2,N}} where {Tv1,Tv2,N}
    project!(
        reshape(CatView(h...), length(h[1]), length(h))', # convert h to a matrix by unrolling the component arrays
        reshape(S∇, length(S∇)), # convert to a vector
        S, Binv=Binv,
    )
    h
end

"""
    projecta!(h::AbstractArray, S∇, S::AbstractMatrix; Binv=I, γ::Integer=5)

Approximate version of `project!` that projects onto each column of `S` separately. Iterates over a random permutation
of the columns `γ` times. This method is equivalent to `project!` if the columns of `S` are orthogonal.
"""
function projecta!(h::AbstractArray, S∇, S::AbstractMatrix; Binv=I, γ::Integer=5)
    γ >= 1 || throw(DomainError(γ, "γ must be positive."))
    p = collect(1:size(S, 1))
    for _ in 1:γ
        shuffle!(p)
        for i in p
            project!(h, selectdim(S∇, 1, i), view(S, :, i), Binv=Binv)
        end
    end
    h
end

"""
    projecta!(h::AbstractArray, S∇::AbstractVector, S::AbstractMatrix; Binv=I, γ::Integer=5)

Approximate version of `project!` for vector gradients.
"""
function projecta!(h::AbstractArray, S∇::AbstractVector, S::AbstractMatrix; Binv=I, γ::Integer=5)
    γ >= 1 || throw(DomainError(γ, "must be positive."))
    p = collect(1:size(S, 1))
    for _ in 1:γ
        shuffle!(p)
        for i in p
            project!(h, S∇[i], view(S, :, i), Binv=Binv)
        end
    end
    h
end

projecta!(h::AbstractArray, S∇, s::AbstractVector; Binv=I, γ::Integer=5) = project!(h, S∇, s; Binv=Binv)

### Biased SEGA

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
    project!(sega::BiasSEGA, args...; kwargs...)

Calls `project!(sega.h, args...; kwargs...)`.

"""
project!(sega::BiasSEGA, args...; kwargs...) = project!(sega.h, args...; kwargs...)
projecta!(sega::BiasSEGA, args...; kwargs...) = projecta!(sega.h, args...; kwargs...)

gradient!(∇, sega::BiasSEGA) = ∇ .= sega.h
gradient(sega::BiasSEGA) = gradient!(zero(sega.h), sega)

### Unbiased SEGA

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
SEGA(θ, h::Array{T,N}) where {T,N} = SEGA{T,N}(θ, deepcopy(h), BiasSEGA(h), deepcopy(h))
function SEGA(θ, hp::Array{T,N}, h::Array{T,N}, g::Array{T,N}) where {T,N} 
    size(hp) == size(h) || throw(DimensionMismatch("hp has dimensions $(size(hp)), h has dimensions $(size(h))"))
    size(g) == size(h) || throw(DimensionMismatch("g has dimensions $(size(g)), h has dimensions $(size(h))"))
    SEGA{T,N}(θ, hp, BiasSEGA(h), g)
end

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
function gradient!(∇::AbstractArray{T,N}, sega::SEGA{T,N}) where {T,N}
    gradient!(∇, sega.h) # store the current (biased) estimate in ∇
    ∇ .*= sega.θ
    ∇ .+= (1-sega.θ) .* sega.hp
end
gradient(sega::SEGA) = gradient!(zero(sega.hp), sega)

end