mutable struct DenseScoreModel <: ScoreModel
    tv_ix::Array{Bool,1}
    no_params::Int
    no_tv::Int
    no_static::Int
    results::ScoreResultsContainer
    criterion::Function
    sdm_criterion::Function
    criterion_reduce::Function
    parameter_transforms::Tuple
    data::CurrentSample
    init_options::InitOptions
    scaling_options::AbstractScalingOptions
end

function get_ωAB(model::DenseScoreModel, sdm_params::Array)
    k=0;
    ω=sdm_params[1:model.no_params]; k+=model.no_params;
    ω=reshape(ω, :, 1)

    ixs=[
        (i,j)
        for i=1:model.no_params
        for j=1:model.no_params
        if model.tv_ix[i]==true & model.tv_ix[j]==true
    ]

    A=spzeros(typeof(ω[1]),model.no_params,model.no_params)
    B=spzeros(typeof(ω[1]),model.no_params,model.no_params)

    o=model.no_tv^2
    for (i,j) in ixs
        k+=1
        A[j,i]=sdm_params[k]
        B[j,i]=sdm_params[k+o]
        if i==j
            A[i,j]=VariableTransforms.from_R_to_pos(A[i,j])
            B[i,j]=VariableTransforms.from_R_to_11(B[i,j])
        end
    end

    ω, A, B
end

function initialize_parameters!(model::DenseScoreModel, result::ScoreResults, ω::Array; A=0.01, B=0.99)
    A_tv=diagm(0=>fill(VariableTransforms.from_pos_to_R(A),model.no_tv))[:]
    B_tv=diagm(0=>fill(VariableTransforms.from_11_to_R(B),model.no_tv))[:]
    parameters!(result,[ω;A_tv;B_tv])
end

get_ωᶜ(model::ScoreModel,ω,B)=(diagm(0=>ones(model.no_params))-B)*ω
get_next_ft(model::ScoreModel,ωᶜ,B,ft,A,Sti,∇t)=ωᶜ+B*ft+A*(Sti*∇t)
