module SelfDrivingFilters
    import Optim
    import LineSearches
    import ForwardDiff
    import DiffResults
    import Base.Iterators
    using LinearAlgebra
    using SparseArrays
    using Statistics: mean
    using VariableTransforms

    export get_ωAB
    export define_model
    export initialize_parameters!
    export initialize_recursion!
    export add_data!
    export fit!
    # export estimate_plots!

    abstract type ScoreFilter end

    # include("variable_transforms.jl")
    include("score_filter_results.jl")
    include("add_data.jl")
    include("score_filter_storage.jl")
    include("scaling.jl")
    include("init_functions.jl")
    include("sparse_score_filters.jl")
    include("simple_score_filters.jl")
    include("dense_score_filters.jl")
    include("leveraged_score_filters.jl")
    include("recursion.jl")
    # include("plotting.jl")
    include("fit.jl")

    function define_model(
            tv_ix::Array{Bool},
            criterion::Function;
            filter_type=:sparse,
            criterion_reduce::Function = mean,
            parameter_transforms=(),
            leverage_ix=Dict{Int,Int}(),
            data=[],
            init_options=(:reverse_run, 1.0),
            scaling_options=(:hessian, 0.99),
            data_type=Float64
        )
        no_params=length(tv_ix)
        if filter_type==:sparse
            model=SparseScoreFilter(data_type,typeof(criterion),typeof(criterion_reduce))
        elseif filter_type==:simple
            model=SimpleScoreFilter(data_type,typeof(criterion),typeof(criterion_reduce))
        elseif filter_type==:dense
            model=DenseScoreFilter(data_type,typeof(criterion),typeof(criterion_reduce))
        elseif filter_type==:leveraged
            model=LeveragedScoreFilter(data_type,typeof(criterion),typeof(criterion_reduce))
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
            model.leverage_ix=AB_ix
        end

        model.criterion=criterion
        model.criterion_reduce=criterion_reduce
        model.tv_ix=tv_ix
        model.no_params = no_params
        model.no_tv = sum(tv_ix)
        model.no_static = model.no_params-model.no_tv
        model.parameter_transforms=parameter_transforms
        if typeof(init_options)!=Symbol
            model.init_options=InitOptions(init_options...)
        else
            model.init_options=InitOptions(init_options)
        end
        if typeof(scaling_options)!=Symbol
            model.scaling_options=ScalingOptions(scaling_options...)
        else
            model.scaling_options=ScalingOptions(scaling_options)
        end
        return model
    end
end # module
