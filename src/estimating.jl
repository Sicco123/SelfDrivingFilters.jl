import Optim

function get_optimizers(criterion, seed; opt=:Newton)
    if opt==:ND || opt==:NelderMead
        return Optim.NelderMead(), criterion
    elseif opt==:BFGS
        return Optim.BFGS(), Optim.OnceDifferentiable(criterion, seed; autodiff=:forward)
    elseif opt==:BFGS_Numerical
        return Optim.BFGS(), Optim.OnceDifferentiable(criterion, seed; autodiff=:finite)
    elseif opt==:LBFGS
        return Optim.LBFGS(), Optim.OnceDifferentiable(criterion, seed; autodiff=:forward)
    elseif opt==:LBFGS_Numerical
        return Optim.LBFGS(), Optim.OnceDifferentiable(criterion, seed; autodiff=:finite)
    elseif opt==:Newton_Numerical
        return Optim.Newton(), Optim.TwiceDifferentiable(criterion, seed; autodiff=:finite)
    elseif opt==:Newton
        return Optim.Newton(), Optim.TwiceDifferentiable(criterion, seed; autodiff=:forward)
    elseif opt==:NewtonTrustRegion_Numerical
        return Optim.NewtonTrustRegion(), Optim.TwiceDifferentiable(criterion, seed; autodiff=:finite)
    elseif opt==:NewtonTrustRegion
        return Optim.NewtonTrustRegion(), Optim.TwiceDifferentiable(criterion, seed; autodiff=:forward)
    end
end

function aicbic(llavg,params)
    Dict(
        :AIC=>2*length(params)-2*size(data)[1]*llavg,
        :BIC=>log(size(data)[1])*length(params)-2*size(data)[1]*llavg
    )
end


function estimate_model(
        criterion,
        parameters,
        model_name;
        optimizers=[
            (:NelderMead, 30),
            (:Newton, 120)
        ],
        model=nothing
    )
    res=0
    for (i,v) in enumerate(optimizers)
        (opt, time_limit) = v
        optimizer, c=get_optimizers(criterion, parameters; opt=opt)
        start_time = [time()]
        res=Optim.optimize(
            c,
            parameters,
            optimizer,
            Optim.Options(
                time_limit=time_limit,
            )
        )
        parameters=res.minimizer
        if i<length(optimizers) && model!=nothing
            parameters!(model.results.initial, res.minimizer)
            initialize_recursion!(model, model.results.initial)
        end
    end
    res
end
function estimate_model_plots(
        criterion,
        parameters,
        model_name;
        optimizers=[
            (:NelderMead, 30),
            (:Newton, 120)
        ],
        model=nothing,
        plot_every=3,
        plots=true,
        reference_paths=[],
        reference_labels=[]
    )
    if plots==true plot_paths_opt(parameters, "Initial", criterion; model_name=model_name,reference_paths=reference_paths,reference_labels=reference_labels) end
    res=0
    for (i,v) in enumerate(optimizers)
        (opt, time_limit) = v
        optimizer, c=get_optimizers(criterion, parameters; opt=opt)
        start_time = [time()]
        res=Optim.optimize(
            c,
            parameters,
            optimizer,
            Optim.Options(
                time_limit=time_limit,
                extended_trace=true,
                store_trace=false,
                callback = res -> optim_plot_update!(res,plot_every,start_time,opt,plots,i,model_name,criterion,reference_paths,reference_labels)
            )
        )
        parameters=res.minimizer
        if i<length(optimizers) && model!=nothing
            parameters!(model.results.initial, res.minimizer)
            initialize_recursion!(model, model.results.initial)
        end
        if plots==true Main.IJulia.clear_output(true) end
    end
    if plots==true plot_paths_opt(res.minimizer, "Finished", criterion; model_name=model_name,reference_paths=reference_paths,reference_labels=reference_labels) end
    res
end


function estimate!(
        model::ScoreModel, initial_parameters::AbstractArray;
        estimate_static=true,
        As=[0.01],
        Bs=[0.99],
        optimizers=[
            (:ND,10),
            (:Newton,120),
        ],
    )
    if estimate_static==true
        st_criterion(parameters; kwargs...) = static_criterion(model)(parameters; kwargs...)
        res=estimate_model(st_criterion, initial_parameters[:], :Static; optimizers=optimizers)
        parameters!(model.results.static, res.minimizer)
        criterion_value!(model.results.static, res.minimum)
        fts=copy(res.minimizer)
        for (i,f) in model.parameter_transforms
            fts[i]=f(fts[i])
        end
        fts=fts .* ones(model.no_params, model.data.T)
        paths!(model.results.static, fts)
        initial_parameters=res.minimizer
    end

    sdm_criterion(parameters; kwargs...) = SelfDrivingScore.sdm_criterion(model)(parameters; kwargs...)
    model.sdm_criterion=sdm_criterion

    ABλ=[
        [A B λ]
        for λ in model.scaling_options.λ
        for A in As
        for B in Bs
    ]

    if length(ABλ)>1
        # println("Searchig for starting values")
        criterion_values=[
            begin
                model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
                initialize_model!(model, model.results.initial, initial_parameters[:]; A=A, B=B)
                sdm_criterion(model.results.initial.parameters[:])
            end
            for (A, B, λ) in ABλ
        ]
        ix = .~isnan.(criterion_values)
        criterion_values=criterion_values[ix]
        ABλ=ABλ[ix]
        ABλ=ABλ[criterion_values.==minimum(criterion_values)]
        # println("Best starting values:")
        # println(ABλ[1])
    end

    (A, B, λ) = ABλ[1]
    model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
    initialize_model!(model, model.results.initial, initial_parameters[:]; A=A, B=B)

    sdm_criterion(model.results.initial.parameters)
    res=estimate_model(
        model.sdm_criterion,
        model.results.initial.parameters[:],
        :SelfDriving;
        optimizers=optimizers,
        model=model
    )
    criterion_value!(model.results.final,res.minimum)
    parameters!(model.results.final, res.minimizer)
    paths!(model.results.final, model.sdm_criterion(res.minimizer;return_paths=true)[:ft])
    init_St!(model.results.final,model.results.initial)
    init_ft!(model.results.final,model.results.initial)
    res
end
function estimate_plots!(
        model::ScoreModel, initial_parameters::AbstractArray;
        estimate_static=true,
        As=[0.01],
        Bs=[0.99],
        optimizers=[
            (:ND,10),
            (:Newton,120),
        ],
        plot_every=3,
        plots=true,
        reference_paths=[],
        reference_labels=[]
    )
    if estimate_static==true
        st_criterion(parameters; kwargs...) = static_criterion(model)(parameters; kwargs...)
        res=estimate_model(st_criterion, initial_parameters[:], :Static; optimizers=optimizers)
        parameters!(model.results.static, res.minimizer)
        criterion_value!(model.results.static, res.minimum)
        fts=copy(res.minimizer)
        for (i,f) in model.parameter_transforms
            fts[i]=f(fts[i])
        end
        fts=fts .* ones(model.no_params, model.data.T)
        paths!(model.results.static, fts)
        initial_parameters=res.minimizer
        reference_paths=[reference_paths...,fts]
        reference_labels=[reference_labels...,:Static]
    end

    sdm_criterion(parameters; kwargs...) = SelfDrivingScore.sdm_criterion(model)(parameters; kwargs...)
    model.sdm_criterion=sdm_criterion

    ABλ=[
        [A B λ]
        for λ in model.scaling_options.λ
        for A in As
        for B in Bs
    ]

    if length(ABλ)>1
        # println("Searchig for starting values")
        criterion_values=[
            begin
                model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
                initialize_model!(model, model.results.initial, initial_parameters[:]; A=A, B=B)
                sdm_criterion(model.results.initial.parameters[:])
            end
            for (A, B, λ) in ABλ
        ]
        ix = .~isnan.(criterion_values)
        criterion_values=criterion_values[ix]
        ABλ=ABλ[ix]
        ABλ=ABλ[criterion_values.==minimum(criterion_values)]
        # println("Best starting values:")
        # println(ABλ[1])
    end

    (A, B, λ) = ABλ[1]
    model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
    initialize_model!(model, model.results.initial, initial_parameters[:]; A=0.0, B=0.0)
    if plots==true plot_paths_opt(model.results.initial.parameters[:], "Initial", model.sdm_criterion; model_name=:SelfDriving,reference_paths=reference_paths,reference_labels=reference_labels) end

    (A, B, λ) = ABλ[1]
    model.scaling_options=ScalingOptions(model.scaling_options.type, λ)
    initialize_model!(model, model.results.initial, initial_parameters[:]; A=A, B=B)

    res=estimate_model_plots(
        model.sdm_criterion,
        model.results.initial.parameters[:],
        :SelfDriving;
        optimizers=optimizers,
        model=model,
        plot_every=plot_every,
        plots=plots,
        reference_paths=reference_paths,
        reference_labels=reference_labels
    )
    criterion_value!(model.results.final,res.minimum)
    parameters!(model.results.final, res.minimizer)
    paths!(model.results.final, model.sdm_criterion(res.minimizer;return_paths=true)[:ft])
    init_St!(model.results.final,model.results.initial)
    init_ft!(model.results.final,model.results.initial)
    res
end
