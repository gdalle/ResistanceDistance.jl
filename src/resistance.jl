using AlgebraicMultigrid
using ChainRulesCore
using Graphs
using Krylov
using LinearAlgebra
using LinearOperators
using ResistanceDistance
using SimpleWeightedGraphs
using Zygote

function laplacian_from_weights(g::AbstractGraph, w::AbstractVector)
    D = Diagonal(degree(g))
    A = adjacency_matrix(g)
    A_up = sparse(UpperTriangular(A))
    W_up = SparseMatrixCSC(A_up.m, A_up.n, A_up.colptr, A_up.rowval, w)
    W = sparse(Symmetric(W_up, :U))
    L = D - W
    return L
end

function resistance_distances(
    g::AbstractGraph, w::AbstractVector{<:Real}, S::AbstractVector{<:Integer}
)
    L = laplacian_from_weights(g, w)
    B = sparse(transpose(incidence_matrix(g; oriented=true)))
    Eₛ = I(nv(g))[:, S]
    L⁺Eₛ = pinv(L) * Eₛ  # L \ Eₛ doesn't work
    R = zeros(eltype(L⁺Eₛ), length(S), length(S))
    R_grads = Matrix{Vector{eltype(L⁺Eₛ)}}(undef, length(S), length(S))
    for (ki, i) in enumerate(S), (kj, j) in enumerate(S)
        R[ki, kj] = L⁺Eₛ[i, ki] - L⁺Eₛ[i, kj] - L⁺Eₛ[j, ki] + L⁺Eₛ[j, kj]
        R_grad_sqrt_ij = B * (L⁺Eₛ[:, ki] - L⁺Eₛ[:, kj])
        R_grads[ki, kj] = -R_grad_sqrt_ij .^ 2
    end
    return R, R_grads
end

function resistance_distances_true(
    g::AbstractGraph, w::AbstractVector{<:Real}, S::AbstractVector{<:Integer}
)
    L = laplacian_from_weights(g, w)
    Γ = pinv(Matrix(L) .+ inv(nv(g)))
    Ω = Matrix{eltype(Γ)}(undef, nv(g), nv(g))
    for i in S, j in S
        Ω[i, j] = Γ[i, i] - Γ[i, j] - Γ[j, i] + Γ[j, j]
    end
    return Ω[S, S]
end

g = SimpleWeightedGraph(Graphs.grid((3, 3)))
S = [1, 2, 3]
w = ones(ne(g))
R = resistance_distances(g, w, S)[1]
R_true = resistance_distances_true(g, w, S)
@assert R ≈ R_true
