function aicbic(llavg,params)
    Dict(
        :AIC=>2*length(params)-2*size(data)[1]*llavg,
        :BIC=>log(size(data)[1])*length(params)-2*size(data)[1]*llavg
    )
end
function fit!(
        model::ScoreFilter, initial_parameters::AbstractArray;
        estimate_static=true,
        As=[0.01],
        Bs=[0.99],
        optimizers=[
            (Optim.NelderMead(),(iterations=100, time_limit=30)),
            (Optim.BFGS(linesearch = LineSearches.BackTracking()),(iterations=1000, time_limit=3600)),
        ],
        autodiff=true
    )
    function get_optimizer(t)
        if length(t)==2
            return t[1], Optim.Options(;t[2]...)
        else
            return t[1], Optim.Options()
        end
    end
    res=nothing
    if estimate_static==true
        st_criterion(parameters; kwargs...) = static_criterion(model)(parameters; kwargs...)
        if autodiff==true
            st_criterion=Optim.TwiceDifferentiable(st_criterion, initial_parameters[:,1]; autodiff = :forward)
        end
        parameters!(model.results.static, initial_parameters[:,1])
        for opt in optimizers
            res=Optim.optimize(st_criterion, parameters(model.results.static), get_optimizer(opt)...)
            parameters!(model.results.static, res.minimizer)
        end
        criterion_value!(model.results.static, res.minimum)
        optim_res!(model.results.static, res)
        fts=copy(res.minimizer)
        for (i,f) in model.parameter_transforms
            fts[i]=f(fts[i])
        end
        fts=fts .* ones(model.no_params, model.data.T+1)
        paths!(model.results.static, fts)
        initial_parameters=res.minimizer
    end

    sd_criterion_function(parameters; kwargs...) = score_driven_criterion(model)(parameters; kwargs...)

    ABλ=Iterators.product(As, Bs, model.scaling_options.λ)
    best_ABλ=((),Inf)
    for (A, B, λ) in ABλ
        model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
        initialize_model!(model, model.results.initial, initial_parameters[:,1]; A=A, B=B)
        o=sd_criterion_function(parameters(model.results.initial))
        if isinf(o) || isnan(o)
            continue
        end
        if o<best_ABλ[2]
            best_ABλ=((A, B, λ), o)
        end
    end
    (A, B, λ) = best_ABλ[1]
    model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
    initialize_model!(model, model.results.initial, initial_parameters[:,1]; A=A, B=B)
    score_driven_criterion!(model, model.results.initial)

    if autodiff==true
        sd_criterion_function=Optim.TwiceDifferentiable(sd_criterion_function, parameters(model.results.initial); autodiff = :forward)
    end
    parameters!(model.results.best, model.results.initial)
    for opt in optimizers
        initialize_model!(model, model.results.best)
        res=Optim.optimize(sd_criterion_function, model.results.best.parameters, get_optimizer(opt)...)
        parameters!(model.results.best, res.minimizer)
    end
    score_driven_criterion!(model, model.results.best)
    optim_res!(model.results.best, res)
end