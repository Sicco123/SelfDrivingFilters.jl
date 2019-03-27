abstract type AbstractScalingOptions end
struct ScalingOptions <: AbstractScalingOptions
    type::Symbol
    λ::Real
end
struct ScalingOptionsRange <: AbstractScalingOptions
    type::Symbol
    λ::AbstractArray
end

function scaling_hessian(model,criterion_t,ft)
    diff_res=DiffResults.HessianResult(ft)
    cfg = ForwardDiff.HessianConfig(criterion_t, diff_res, ft, ForwardDiff.Chunk{model.no_params}())
    diff_res_function(ft)=ForwardDiff.hessian!(diff_res, criterion_t, ft, cfg)
    get_St_∇t(diff_res, Ht, Gt, St, λ, Imλ)=begin
        ∇t=DiffResults.gradient(diff_res)
        Ht=λ*Ht+Imλ*DiffResults.hessian(diff_res)
        St=Ht
        return Ht, Gt, St, ∇t
    end
    return diff_res, diff_res_function, get_St_∇t
end
function scaling_OPG(model,criterion_t,ft)
    diff_res=DiffResults.GradientResult(ft)
    cfg = ForwardDiff.GradientConfig(criterion_t, ft, ForwardDiff.Chunk{model.no_params}())
    diff_res_function(ft)=ForwardDiff.gradient!(diff_res, criterion_t, ft, cfg)
    get_St_∇t(diff_res, Ht, Gt, St, λ, Imλ)=begin
        ∇t=DiffResults.gradient(diff_res)
        Gt=λ*Gt+Imλ*(∇t*∇t')
        St=Gt
        return Ht, Gt, St, DiffResults.gradient(diff_res)
    end
    return diff_res, diff_res_function, get_St_∇t
end
function scaling_robust(model,criterion_t,ft)
    diff_res=DiffResults.HessianResult(ft)
    cfg = ForwardDiff.HessianConfig(criterion_t, diff_res, ft, ForwardDiff.Chunk{model.no_params}())
    diff_res_function(ft)=ForwardDiff.hessian!(diff_res, criterion_t, ft, cfg)
    get_St_∇t(diff_res, Ht, Gt, St, λ, Imλ)=begin
        ∇t=DiffResults.gradient(diff_res)
        Gt=λ*Gt+Imλ*(∇t*∇t')
        Ht=λ*Ht+Imλ*DiffResults.hessian(diff_res)
        try
            St=Ht*inv(Gt)*Ht
        catch
            St=Ht*pinv(Gt)*Ht
        end
        return Ht, Gt, St, ∇t
    end
    return diff_res, diff_res_function, get_St_∇t
end
function scaling_unit(model,criterion_t,ft)
    diff_res=DiffResults.GradientResult(ft)
    cfg = ForwardDiff.GradientConfig(criterion_t, ft, ForwardDiff.Chunk{model.no_params}())
    diff_res_function(ft)=ForwardDiff.gradient!(diff_res, criterion_t, ft, cfg)
    get_St_∇t(diff_res, Ht, Gt, St, λ, Imλ)=begin
        ∇t=DiffResults.gradient(diff_res)
        return Ht, Gt, St, DiffResults.gradient(diff_res)
    end
    return diff_res, diff_res_function, get_St_∇t
end
function scaling_outer(model,criterion_t,ft)
    if model.scaling_options.type==:hessian
        return scaling_hessian(model,criterion_t,ft)
    elseif model.scaling_options.type==:opg
        return scaling_OPG(model,criterion_t,ft)
    elseif model.scaling_options.type==:robust
        return scaling_robust(model,criterion_t,ft)
    elseif model.scaling_options.type==:unit
        return scaling_unit(model,criterion_t,ft)
    end
end
