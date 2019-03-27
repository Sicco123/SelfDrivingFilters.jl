mutable struct SparseScoreModel <: ScoreModel
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

function get_ωAB(model::SparseScoreModel, sdm_params::Array)
    k=0;
    ω=sdm_params[1:model.no_params]; k+=model.no_params;
    ω=reshape(ω, :, 1)

    A_tv=VariableTransforms.from_R_to_pos.(sdm_params[1+k:k+model.no_tv]); k+=model.no_tv;
    B_tv=VariableTransforms.from_R_to_11.(sdm_params[1+k:k+model.no_tv])

    A=zeros(typeof(ω[1]), model.no_params)
    A[model.tv_ix]=A_tv[:]

    B=zeros(typeof(ω[1]), model.no_params)
    B[model.tv_ix]=B_tv[:]

    ω, A, B
end

function initialize_parameters!(model::SparseScoreModel, result::ScoreResults, ω::Array; A=0.01, B=0.99)
    A_tv=VariableTransforms.from_pos_to_R.(fill(A, model.no_tv, 1))
    B_tv=VariableTransforms.from_11_to_R.(fill(B, model.no_tv, 1))
    parameters!(result,[ω;A_tv;B_tv])
end

get_ωᶜ(model::SparseScoreModel,ω,B)=(1.0.-B).*ω
get_next_ft(model::SparseScoreModel,ωᶜ,B,ft,A,Sti,∇t)=ωᶜ.+B.*ft.+A.*(Sti*∇t)
