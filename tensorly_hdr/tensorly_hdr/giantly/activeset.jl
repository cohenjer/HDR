# NNLS solver from Portugal 1994, "A comparison on block pivoting [...]"

export activeset, support_activeset, matrixactiveset

function activeset(A::AbstractMatrix{T},
                   b::AbstractVector{T},
                   x0::AbstractVector{T}=Vector{T}(undef, 0);
                   inputgram::Bool=false,
                   outerror::Bool=false,
                   btb::T=0.0, # btb is used only if outerror is true
                   sumtoone::Bool=false,
                   tol::T=1e-12
                   ) where T <: AbstractFloat
    # Init Gram matrices if needed
    AtA = inputgram ? A : A'*A
    Atb = inputgram ? b : A'*b

    # Init constants
    r = size(AtA, 1)
    maxiter = 5*r

    # Init variables
    t = r + 1
    p = 3

    # If not provided, init x full of zeros
    x = (length(x0) == 0) ? zeros(r) : x0
    y = zeros(r)

    # Init support (what Kim calls passive set, ie, entries not constrained to 0)
    F = x .> tol

    # NNLS main loop
    for it in 1:maxiter
        # Solve least squares (x) and compute gradient (y)
        if !sumtoone # standard algorithm
            x[F] = AtA[F,F] \ Atb[F]
            y[.~F] = AtA[.~F,F] * x[F] - Atb[.~F]
        else # modified so that solution has unit l1 norm
            sF = sum(F)
            As = [AtA[F,F] -ones(sF,1) ; ones(1,sF) 0]
            bs = [Atb[F] ; 1]
            sol = As \ bs
            x[F] = sol[1:sF]
            mu = sol[sF+1]
            y[.~F] = AtA[.~F,F] * x[F] - Atb[.~F] .- mu
        end
        x[.~F] .= 0
        y[F] .= 0

        # Compute h1 and h2 (optimality conditions)
        h1 = (x .< -tol) .& F
        h2 = (y .< -tol) .& .~F
        h1c = falses(r)
        h2c = falses(r)

        notgood = sum(h1) + sum(h2)

        # Stopping criterion (if optimality reached)
        if notgood <= 0
            break
        end

        # Update
        if notgood < t
            t = notgood
            p = 3
            h1c = copy(h1)
            h2c = copy(h2)
        end

        if (notgood >= t) & (p >= 1)
            p = p - 1
            h1c = copy(h1)
            h2c = copy(h2)
        end

        if (notgood >= t) & (p == 0)
            tochange = findlast(h1 .| h2)
            if tochange!=nothing && h1[tochange]
                h1c .= false
                h1c[tochange] = true
                h2c .= false
            else
                h1c .= false
                h2c .= false
                if tochange!=nothing
                    h2c[tochange] = true
                end
            end
        end

        F = xor.(F, h1c) .| h2c
    end
    # Remove numeric noise
    x[x .< tol] .= 0
    # Return solution, and if needed also return the residual error
    if outerror
        if iszero(btb)
            error("Keyword argument btb is needed to output the error")
            # TODO: procs if a column is entirely zero. 
            # If this happens in practice, catch that case
        end
        return x, (btb - Atb[F]' * x[F])
    else
        return x
    end
end


# Solve NNLS constrained to a given support
# This function was intended for use only with the branch-and-bound algo
# so it accepts only the Gram matrices in input
function support_activeset(AtA::AbstractMatrix{T},
                           Atb::AbstractVector{T},
                           btb::T,
                           supp::Vector{Int64},
                           x0::Vector{T}=Vector{T}(undef, 0);
                           sumtoone::Bool=false
                           ) where T <: AbstractFloat
    # Init constant
    r = size(AtA, 1)
    # If not provided, init x0 full of zeros
    x0 = (length(x0) == 0) ? zeros(r) : x0
    # Compute x with activeset restricted to the given support
    x = zeros(r)
    x[supp], resid = activeset(AtA[supp, supp], Atb[supp], x0[supp],
                               inputgram=true, outerror=true, btb=btb,
                               sumtoone=sumtoone)
    # Return solution and residual error
    return x, resid
end


# Solve multiple right-hand side NNLS
function matrixactiveset(X::AbstractMatrix{T},
                         W::AbstractMatrix{T},
                         H0::AbstractMatrix{T}=Matrix{T}(undef, 0, 0);
                         inputgram::Bool=false,
                         sumtoone::Bool=false
                         ) where T <: AbstractFloat
    # If not provided, precompute Gram matrices
    WtW = inputgram ? W : W' * W
    WtX = inputgram ? X : W' * X
    # Init constants
    r, n = size(WtX)
    # If not provided, init H full of zeros
    H = length(H0) == 0 ? zeros(r, n) : H0
    # Solve the n NNLS subproblems with activeset
    for j in 1:n
        H[:,j] .= activeset(WtW, WtX[:,j], H[:,j], sumtoone=sumtoone)
    end
    # Return
    return H
end
