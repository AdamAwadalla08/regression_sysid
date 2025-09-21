using Random

function generate_polynomial_data(n::Int,degree::Int;
                                x_range=(-3.0:3.0),
                                sigma = 2.0,
                                hetero::Bool=false,
                                seed=nothing,
                                beta=nothing,
                                misspec_bump::Int=0)

    if seed!==nothing
        Random.seed!(seed)
    end

    xmin,xmax = x_range
    x = range(xmin,xmax;length=n)  |> collect

    if beta === nothing
        beta_true = [randn() for _ in 0:degree]

        for k in 2:length(beta_true)
            beta_true[k] /=max(1,k-1)
        end
    
    else
        @assert length(beta) == degree+1 "polynomial coefficients must have length of degree + 1"
        beta_true = collect(beta)
    end

    # Optional misspecification: (tiny degree bump)
    delta = 0.0
    if misspec_bump > 0
        deg_extra = degree+misspec_bump
    c_extra = 0.02 * randn()
    delta = c_extra.*x.^deg_extra
    end

    # build y (without noise or misspec)
    y_clean = similar(x,Float64)
    y_clean .= 0.0

    for (k,bk) in enumerate(beta_true)
        @inbounds y_clean .+= bk.*(x.^(k-1))
    end
    
    #generate noise based on std.
    if hetero
    sig_x = sigma .*(1.0.+0.5.*abs.(x))
    noise = sig_x.*randn(n)
    else
        noise = sigma .*randn(n)
    end
    
    y = y_clean.+noise
    
    return (x=x,y=y,y_clean=y_clean,noise=noise,beta_true=beta_true)
end