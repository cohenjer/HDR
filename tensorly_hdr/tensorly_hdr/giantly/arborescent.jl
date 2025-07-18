using LinearAlgebra

export arborescent, ksparse_mnnls, ksparse_updtH!


mutable struct BabNode
    sol::Vector{Float64}
    resid::Float64
end

# Sparse NNLS solver for one vector
function arborescent(AtA, Atb, btb, k; returnpareto=false, returnnbnodes=false, sumtoone=false)
    r = size(AtA, 1)

    # Init values
    support = collect(1:r)
    nbnodes = [1] # Number of nodes explored
    bze = 1 # biggest zeroed entry (to avoid symmetry in branching)
    bestresid = norm(Atb)
    # Init pareto front
    paretofront = BabNode[]
    for _ in 1:r-k
        push!(paretofront, BabNode(zeros(r), Inf))
    end

    # Root node (unconstrained nnls)
    bestx = activeset(AtA, Atb, inputgram=true, sumtoone=sumtoone)
    prevx = copy(bestx)

    # Sort support so that the smallest entries of x are constrained first
    sort!(support, by=i->bestx[i])

    # Call to recursive branch-and-bound
    bestx, bestresid = bab(AtA, Atb, btb, k, support, prevx, bze,
                           bestx, bestresid, paretofront, nbnodes,
                           sumtoone=sumtoone)

    if returnpareto
        paretomat = zeros(r, r-k)
        for j in 1:r-k
            paretomat[:,j] .= paretofront[r-k+1-j].sol
        end
        if returnnbnodes
            return hcat(bestx, paretomat), nbnodes[1]
        else
            return hcat(bestx, paretomat)
        end
    else
        if returnnbnodes
            return bestx, nbnodes[1]
        else
            return bestx
        end
    end
end



function bab(AtA, Atb, btb, k, support, prevx, bze, bestx, bestresid, paretofront, nbnodes; sumtoone=false)
    # Get dimensions
    r = size(AtA, 2)
    kprime = length(support)

    # Compute nnls with current support
    x, resid = support_activeset(AtA, Atb, btb, support, prevx; sumtoone=sumtoone)
    nbnodes[1] += 1

    # If current residual is worst than bound, then prune
    if resid >= bestresid
        return bestx, bestresid
    end

    # If sparsity k is reached and residual is better than bound, then return new best sol
    if kprime <= k
        return x, resid
    end

    # Test current k'-sparse best sol and potentially update
    if resid < paretofront[r-kprime+1].resid
        paretofront[r-kprime+1].resid = resid
        paretofront[r-kprime+1].sol .= x
    end

    # If sparsity k is not reached and residual is better than bound, then continue
    # One branch for every remaining entry of the support
    for i in bze:length(support)
        inactive = support[i]
        deleteat!(support, i)
        childbze = max(i, bze)
        bestx, bestresid = bab(AtA, Atb, btb, k, support, x, childbze,
                               bestx, bestresid, paretofront, nbnodes,
                               sumtoone=sumtoone)
        insert!(support, i, inactive)
    end

    return bestx, bestresid
end


# k-sparse NNLS solver for multiple right-hand sides
# updates H in place
function ksparse_mnnls(UtM, UtU, MtM, V, k; sumtoone=false)
    # V has shapes r x n
    # Loop on all columns of M and H
    for j in 1:size(V, 2)
        out = arborescent(UtU, UtM[:, j], MtM[j], k, sumtoone=sumtoone)
        V[:, j] .= out
    end
    return
end
