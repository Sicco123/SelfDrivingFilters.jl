mutable struct LeveragedScoreModel <: ScoreModel
    tv_ix::Array{Bool,1}
    leverage_ix::Array{Bool,2}
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

function get_ωAB(model::LeveragedScoreModel, sdm_params::Array)
    k=0;
    ω=sdm_params[1:model.no_params]; k+=model.no_params;
    ω=reshape(ω, :, 1)

    A=spzeros(typeof(ω[1]),model.no_params,model.no_params)
    B=spzeros(typeof(ω[1]),model.no_params,model.no_params)

    o=sum(model.leverage_ix)
    A[model.leverage_ix]=sdm_params[k+1:k+o]; k+=o
    B[model.leverage_ix]=sdm_params[k+1:k+o]

    for (i,b) in enumerate(model.tv_ix)
        if b==true
            A[i,i]=VariableTransforms.from_R_to_pos(A[i,i])
            B[i,i]=VariableTransforms.from_R_to_11(B[i,i])
        end
    end

    ω, A, B
end

function initialize_parameters!(model::LeveragedScoreModel, result::ScoreResults, ω::Array; A=0.01, B=0.99)
    A_tv=diagm(0=>fill(VariableTransforms.from_pos_to_R(A),model.no_params))[model.leverage_ix][:]
    B_tv=diagm(0=>fill(VariableTransforms.from_11_to_R(B),model.no_params))[model.leverage_ix][:]
    parameters!(result,[ω;A_tv;B_tv])
end
