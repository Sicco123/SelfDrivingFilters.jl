struct InitOptions
    type::Symbol
    frac::Number
    InitOptions(t)=new(t,1.0)
    InitOptions(t,f)=new(t,f)
end
type(x::InitOptions)=x.type


function init_scaling_hessian(model, ω, T)
    if length(ω)<model.data.T
        ω=ω.*ones(length(ω), T)
    end
    St=zeros(size(ω)[1],size(ω)[1])

    data=model.data.data
    x=data[:,1]
    criterion_t(ω)=model.criterion(ω, x)
    diff_res, diff_res_function, get_St_∇t = scaling_hessian(model,criterion_t,ω[:,1])
    for i=1:T
        x.=data[:,i]
        diff_res_function(ω[:,i])
        St.+=DiffResults.hessian(diff_res)
    end
    St=St./T
    St[(abs.(St).<=1e-8)].=0.0
    St
end

function init_scaling_OPG(model, ω, T)
    if length(ω)<model.data.T
        ω=ω.*ones(length(ω), T)
    end
    St=zeros(size(ω)[1],T)

    data=model.data.data
    x=data[:,1]
    criterion_t(ω)=model.criterion(ω, x)
    diff_res, diff_res_function, get_St_∇t = scaling_OPG(model,criterion_t,ω[:,1])
    for i=1:T
        x.=data[:,i]
        diff_res_function(ω[:,i])
        St[:,i].=DiffResults.gradient(diff_res)
    end
    St=(St*St')./T
    St[(abs.(St).<=1e-8)].=0.0
    St
end

function init_scaling_robust(model, ω, T)
    if length(ω)<model.data.T
        ω=ω.*ones(length(ω), T)
    end
    Ht=zeros(size(ω)[1],size(ω)[1])
    ∇t=zeros(size(ω)[1],T)

    data=model.data.data
    x=data[:,1]
    criterion_t(ω)=model.criterion(ω, x)
    diff_res, diff_res_function, get_St_∇t = scaling_robust(model,criterion_t,ω[:,1])
    for i=1:T
        x.=data[:,i]
        diff_res_function(ω[:,i])
        Ht.+=DiffResults.hessian(diff_res)
        ∇t[:,i].=DiffResults.gradient(diff_res)
    end
    ∇t=Hermitian((∇t*∇t')/T)
    Ht=Hermitian(Ht/T)
    St=Ht*inv(∇t)*Ht
    St[(abs.(St).<=1e-8)].=0.0
    St
end
function init_scaling_unit(model, ω, T)
    # diagm(0=>ones(model.no_params))
    spdiagm(0=>ones(model.no_params))
end
function init_scaling_outer(model)
    if model.scaling_options.type==:hessian
        return init_scaling_hessian
    elseif model.scaling_options.type==:opg
        return init_scaling_OPG
    elseif model.scaling_options.type==:robust
        return init_scaling_robust
    elseif model.scaling_options.type==:unit
        return init_scaling_unit
    end
end

function initialize_recursion!(model::ScoreFilter, result::ScoreFilterResults)
    init_frac=model.init_options.frac
    T=round(Int, init_frac)
    if init_frac<=1.0
        T=round(Int, init_frac*model.data.T)
    end
    init_ft!(result, result.parameters[1:model.no_params])
    if type(model.init_options)==:static
        # try
        #     init_ft!(result, model.results.static.parameters)
        # catch
        # end
        St=init_scaling_outer(model)(model, result.parameters[1:model.no_params], T)
        init_St!(result, St)
    else
        init_St!(result, diagm(0=>ones(model.no_params)))
        _, fts, St = recursion(model, result, get_ωAB(model, result.parameters)...,T:-1:2; increment=-)
        init_ft!(result, fts[:,1])
        init_St!(result, St)
    end
end

function initialize_model!(model::ScoreFilter, result::ScoreFilterResults, ω::AbstractArray; A=0.01, B=0.99)
    initialize_parameters!(model, result, ω; A=A, B=B)
    initialize_recursion!(model, result)
end
function initialize_model!(model::ScoreFilter, result::ScoreFilterResults)
    initialize_recursion!(model, result)
end