# function static_criterion(model::ScoreFilter, params::AbstractArray{S,1}) where {S<:Number}
#     data=model.data.data
#     T=model.data.T
#     for i=1:model.data.T
#         model.storage.criterion_values[i]=model.criterion(params, view(data,:,i))
#     end
#     model.criterion_reduce(model.storage.criterion_values)::S
# end
function static_criterion(model::ScoreFilter, params::AbstractArray{S,1}) where {S<:Number}
    data=model.data.data
    T=model.data.T
    criterion_values=Array{S}(undef, model.data.T)
    for i=1:model.data.T
        criterion_values[i]=model.criterion(params, view(data,:,i))
    end
    model.criterion_reduce(criterion_values)::S
end
function static_criterion(model::ScoreFilter)
    return criterion(params::Array{T,1}) where {T<:Number}=static_criterion(model,params)
end

function recursion(
        model::ScoreFilter,
        result::ScoreFilterResults,
        ω::AbstractArray{S},
        A::AbstractArray{S},
        B::AbstractArray{S},
        rng::AbstractRange;
        increment = +
    ) where {S<:Number}
    St=convert(Array{S,2},copy(result.init_St));
    Ht=copy(St);
    Gt=copy(St);
    data=model.data.data
    T=model.data.T
    no_params=model.no_params

    fts=Array{S,2}(undef, no_params, T+1);
    fts[:,first(rng)]=result.init_ft.*one(S)


    ωᶜ=get_ωᶜ(model,ω,B)

    x=data[:,first(rng)]
    criterion_t(ft)=model.criterion(ft, x)

    diff_res, diff_res_function!, get_St_∇t! = scaling_outer(model,criterion_t,view(fts,:,first(rng)))
    # criterion_values=Array{S,1}(undef, model.data.T)
    criterion_values=fill(zero(S), model.data.T)

    Sti=similar(St)
    ∇t=similar(ω)
    λ=model.scaling_options.λ::Float64
    Imλ=1-λ
    for i in rng
        x.=data[:,i]
        diff_res_function!(view(fts,:,i))
        get_St_∇t!(diff_res, Ht, Gt, St, λ, Imλ, ∇t)

        criterion_values[i]=DiffResults.value(diff_res)
        if abs(criterion_values[i])>1e7
            break
        end
        Sti.= .-St
        get_next_ft(model,i,increment(i,1),ωᶜ,B,fts,A,Sti,∇t)
    end
    criterion_values, fts, St
end

function score_driven_criterion(model::ScoreFilter, sdm_params::Array; stage=:initial)
    ω, A, B = get_ωAB(model, sdm_params)
    # if typeof(sdm_params[1])==Float64
    #     ix = .~ model.tv_ix
    #     getfield(model.results, stage).init_ft[ix]=ω[ix]
    # end
    criterion_values, fts, _ = recursion(model, getfield(model.results, stage), ω, A, B, 1:model.data.T)
    model.criterion_reduce(criterion_values)
end
function score_driven_criterion(model::ScoreFilter)
    return criterion(sdm_params::Array; kwargs...)=score_driven_criterion(model, sdm_params; kwargs...)
end
function score_driven_criterion!(model::ScoreFilter, result::ScoreFilterResults)
    ω, A, B = get_ωAB(model, parameters(result))
    # if typeof(sdm_params[1])==Float64
    #     ix = .~ model.tv_ix
    #     getfield(model.results, stage).init_ft[ix]=ω[ix]
    # end
    criterion_values, fts, _ = recursion(model, result, ω, A, B, 1:model.data.T)
    o=model.criterion_reduce(criterion_values)

    for (i,f) in model.parameter_transforms
        fts[i,:]=f.(fts[i,:])
    end
    criterion_value!(result,o)
    paths!(result, fts)
    # init_St!(model.results.best,getfield(model.results, stage))
    # init_ft!(model.results.best,getfield(model.results, stage))
end



