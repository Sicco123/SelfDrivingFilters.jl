mutable struct SimpleScoreFilter{T<:Real,F1<:Function,F2<:Function} <: ScoreFilter
    tv_ix::Array{Bool,1}
    no_params::Int
    no_tv::Int
    no_static::Int
    results::ScoreFilterResultsContainer{T}
    criterion::F1
    criterion_reduce::F2
    parameter_transforms::Tuple
    data::CurrentSample{T}
    init_options::InitOptions
    scaling_options::AbstractScalingOptions
    storage::ScoreFilterStorage{T}
    SimpleScoreFilter(S,S1,S2)=new{S,S1,S2}()
end

function get_ωAB(model::SimpleScoreFilter, sdm_params::Array{S,1}) where {S<:Number}
    k=0;
    ω=sdm_params[1:model.no_params]; k+=model.no_params;
    ω=reshape(ω, :, 1)

    A_tv=from_R_to_pos.(sdm_params[k+1])
    B_tv=from_R_to_11.(sdm_params[k+2])

    # A=zeros(S, model.no_params)
    A=spzeros(S, model.no_params)
    A[model.tv_ix].=A_tv

    # B=zeros(S, model.no_params)
    B=spzeros(S, model.no_params)
    B[model.tv_ix].=B_tv

    ω, A, B
end

function initialize_parameters!(model::SimpleScoreFilter, result::ScoreFilterResults, ω::AbstractArray; A=0.01, B=0.99)
    A_tv=from_pos_to_R(A)
    B_tv=from_11_to_R(B)
    parameters!(result,[ω;A_tv;B_tv])
end

get_ωᶜ(model::SimpleScoreFilter,ω,B)=(1.0.-B).*ω
get_next_ft(model::SimpleScoreFilter,i,ip1,ωᶜ,B,fts,A,Sti,∇t)=fts[:,ip1:ip1] .= ωᶜ.+B.*view(fts,:,i).+A.*(Sti\∇t)
