function resistance_solver(L, B::AbstractMatrix; kwargs...)
    A = LinearOperator(L) + opOnes(eltype(L), size(L)...)
    C = block_minres(A, B; kwargs...)[1]
    return C
end

function ChainRulesCore.rrule(::typeof(resistance_solver), L, B::AbstractMatrix; kwargs...)
    A = LinearOperator(L) + opOnes(eltype(L), size(L)...)
    Aᵀ = transpose(A)
    C = block_minres(A, B; kwargs...)[1]
    function resistance_solver_pullback(dC)
        dB = block_minres(A, dC; kwargs...)[1]
        dAᵀ = block_minres(Aᵀ, dB; kwargs...)[1]
        dA = transpose(dAᵀ)
        dL = dA
        return (NoTangent(), dL, dB)
    end
    return C, resistance_solver_pullback
end
