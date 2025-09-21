using Statistics, LinearAlgebra, Polynomials

function linearlsqclosedform(x,y)
    #=
    This is a least squares function using the closed form solution y = b_0 +b_1 * x
    I used this to familiarise myself with some functions
    =#
    xbar = sum(x)/length(x)
    ybar = sum(y)/length(y)

    
    Sxx = sum((x.-xbar).^2)
    Sxy = sum((x.-xbar).*(y.-ybar))
    
    b1 = Sxy/Sxx
    b0 = ybar-b1*xbar

    return b0,b1, b1*x.+b0
end


function linearlsqmatrixform(x,y)
    #=
    this is the same one as the above, but using the matrix form solution of a y = b_0 +b_1 * x
    I did this to get familiar with matrix ops.
    =#
    X = ones(length(x),2)
    X[:,2] = x
    
    b =  (transpose(X)*X)\(transpose(X)*y)
    return b[1], b[2]
end


function lstsqsolver(x,y,order)
    #=
    This is an Nth-order polynomial least squares solver. I used this to get more familiar with
    matrix operations for matrices I don't know the size of.
    solves in form of y = b_0 + b_1*x + b_2*x^2 + ... b_n*x^n 
    =#

    X = x .^ (0:order)' # vandermonde design matrix
    β = (transpose(X)*X) \ (transpose(X)*y) # β = (X^T X)^-1 (X^T y)
    y_ = X*β
    residuals = y .- y_
    ssres = sum(residuals.^2)
    sstot = sum((y.-mean(y)).^2)

    return β, y_, residuals, 1.0 - ssres/sstot 
end


function legendre_vandermonde(x,m)
    n = length(x)
    T = eltype(x)
    X = zeros(n,m+1)
    X[:,1] .= one(T)
    if m >=1
        X[:,2] .= x
        for k in 1:m-1
            X[:,k+2].=((2k+1).*x.*X[:,k+1].-k.*X[:,k])./(k+1)
        end
    end
    return X
end

function scale_to_minus1_1(x::AbstractVector)
    xmin = minimum(x); xmax = maximum(x)
    dx = xmax - xmin
    dx == 0 && throw(ArgumentError("All x are equal; cannot scale to [-1,1]."))
    xscaled = @. 2*(x - xmin)/dx - 1
    return xscaled
end


function wlstsqsolver(x,y,order;weighing_fn="gaussian")
    x = scale_to_minus1_1(x)

    if weighing_fn == "gaussian"
    X = legendre_vandermonde(x,order) # vandermonde design matrix
    w = ones(length(x))
    W = Diagonal(w.^0.5)
    end

    if weighing_fn == "lorentz"
    X = legendre_vandermonde(x,order) # vandermonde design matrix
    w = 1 ./(1 .+x.^2)
    W = Diagonal(w)
    end


    β = (transpose(X)*W*X) \ (transpose(X)*W*y) # β = (X^T X)^-1 (X^T y)
    y_ = X*β
    residuals = y .- y_
    ssres = sum(residuals.^2)
    sstot = sum((y.-mean(y)).^2)

    return β, y_, residuals, 1.0 - ssres/sstot 
end


