function static_criterion(params::Array, model::ScoreModel)
    o=[model.criterion(params, model.data.data[i,:]) for i=1:model.data.T]
    o=model.criterion_reduce(o)
    o
end
function static_criterion(model::ScoreModel)
    return criterion(params::Array)=static_criterion(params, model)
end

function sdm_recursion(
        model::ScoreModel,
        result::ScoreResults,
        ω::AbstractArray,
        A::AbstractArray,
        B::AbstractArray,
        rng::AbstractRange;
    )
    St=result.init_St; Ht=result.init_St; Gt=result.init_St;
    ft=result.init_ft.*one(ω[1,1])

    fts=fill(zero(ω[1]), model.no_params, model.data.T);

    ωᶜ=get_ωᶜ(model,ω,B)

    i=1
    x=model.data.data[i,:]
    criterion_t(ft)=model.criterion(ft, x)

    diff_res, diff_res_function, get_St_∇t = scaling_outer(model,criterion_t,ft)
    criterion_values=fill(zero(criterion_t(ft)), model.data.T)

    Sti=I
    λ=model.scaling_options.λ
    Imλ=1-λ
    for i in rng
        fts[:,i]=ft

        x=model.data.data[i,:]
        diff_res=diff_res_function(ft)
        Ht, Gt, St, ∇t = get_St_∇t(diff_res, Ht, Gt, St, λ, Imλ)

        criterion_values[i]=DiffResults.value(diff_res)
        if abs(criterion_values[i])>1e7
            break
        end
        # # Bias corrected
        # Stu=St/(1-λ^i)

        Stu=-Symmetric(St)
        try
            Sti=inv(Stu)
        catch
            Sti=pinv(Stu)
        end
        ft=get_next_ft(model,ωᶜ,B,ft,A,Sti,∇t)

    end
    criterion_values, fts, St
end

function sdm_criterion(model::ScoreModel, sdm_params::Array; return_paths=false, stage=:initial)
    ω, A, B = get_ωAB(model, sdm_params)
    if typeof(sdm_params[1])==Float64
        ix = .~ model.tv_ix
        getfield(model.results, stage).init_ft[ix]=ω[ix]
    end


    criterion_values, fts, _ = sdm_recursion(model, getfield(model.results, stage), ω, A, B, 1:model.data.T)
    o=model.criterion_reduce(criterion_values)
    if return_paths==true
        for (i,f) in model.parameter_transforms
            fts[i,:]=f.(fts[i,:])
        end
        return Dict(
            :ft=>fts,
            :criterion_value=>o
        )
    end
    if typeof(sdm_params[1])==Float64 && o<model.results.best.criterion_value
        criterion_value!(model.results.best,o)
        parameters!(model.results.best, sdm_params)
        init_St!(model.results.best,getfield(model.results, stage))
        init_ft!(model.results.best,getfield(model.results, stage))
    end
    o
end
function sdm_criterion(model::ScoreModel)
    return criterion(sdm_params::Array; kwargs...)=sdm_criterion(model, sdm_params; kwargs...)
end
