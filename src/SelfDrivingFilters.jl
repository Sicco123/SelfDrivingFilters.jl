module SelfDrivingFilters
    import ForwardDiff
    import DiffResults
    using LinearAlgebra
    using SparseArrays
    using Statistics: mean

    export get_ωAB
    export define_model
    export initialize_parameters!
    export initialize_recursion!
    export add_data!
    export estimate!
    export estimate_plots!

    abstract type ScoreModel end

    include("variable_transforms.jl")
    include("add_data.jl")
    include("score_results.jl")
    include("scaling.jl")
    include("init_functions.jl")
    include("sparse_score_models.jl")
    include("dense_score_models.jl")
    include("leveraged_score_models.jl")
    include("sdm_recursion.jl")
    include("plotting.jl")
    include("estimating.jl")

    function define_model(
            tv_ix::Array{Bool},
            criterion::Function;
            type=:sparse,
            criterion_reduce::Function = mean,
            parameter_transforms=(),
            leverage_ix=Dict{Int,Int}(),
            data=[],
            init_options=InitOptions(:reverse_run, 1.0),
            scaling_options=ScalingOptions(:hessian, 0.99)
        )
        no_params, no_tv, = length(tv_ix), sum(tv_ix);
        no_static = no_params-no_tv;
        if type==:sparse
            model=SparseScoreModel(
                tv_ix,
                no_params,
                no_tv,
                no_static,
                ScoreResultsContainer(ScoreResults(:static,[],NaN,[],[],[]),ScoreResults(:initial,[],NaN,[],[],[]),ScoreResults(:best,[],Inf,[],[],[]),ScoreResults(:final,[],NaN,[],[],[])),
                criterion,
                x->x,
                criterion_reduce,
                parameter_transforms,
                CurrentSample([],0,0),
                init_options,
                scaling_options
            )
        elseif type==:leveraged
            AB_ix=fill(false,no_params,no_params)
            for i=1:no_params
                if tv_ix[i]==true
                    AB_ix[i,i]=true
                end
            end
            for i in keys(leverage_ix)
                for j in leverage_ix[i]
                    AB_ix[j,i]=true
                end
            end
            model=LeveragedScoreModel(
                tv_ix,
                AB_ix,
                no_params,
                no_tv,
                no_static,
                ScoreResultsContainer(ScoreResults(:static,[],NaN,[],[],[]),ScoreResults(:initial,[],NaN,[],[],[]),ScoreResults(:best,[],Inf,[],[],[]),ScoreResults(:final,[],NaN,[],[],[])),
                criterion,
                x->x,
                criterion_reduce,
                parameter_transforms,
                CurrentSample([],0,0),
                init_options,
                scaling_options
            )
        else
            model = DenseScoreModel(
                tv_ix,
                no_params,
                no_tv,
                no_static,
                ScoreResultsContainer(ScoreResults(:static,[],NaN,[],[],[]),ScoreResults(:initial,[],NaN,[],[],[]),ScoreResults(:best,[],Inf,[],[],[]),ScoreResults(:final,[],NaN,[],[],[])),
                criterion,
                x->x,
                criterion_reduce,
                parameter_transforms,
                CurrentSample([],0,0),
                init_options,
                scaling_options
            )
        end
        return model
    end
end # module
