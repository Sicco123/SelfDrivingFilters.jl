mutable struct DenseScoreFilter{T<:Real,F1<:Function,F2<:Function} <: ScoreFilter
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
    DenseScoreFilter(S,S1,S2)=new{S,S1,S2}()
end

function get_ωAB(model::DenseScoreFilter, sdm_params::Array{S,1}) where {S<:Number}
    k=0;
    ω=sdm_params[1:model.no_params]; k+=model.no_params;
    ω=reshape(ω, :, 1)

    A=spzeros(S,model.no_params,model.no_params)
    B=spzeros(S,model.no_params,model.no_params)

    o=model.no_tv^2
    for i=1:model.no_params
        for j=1:model.no_params
            if model.tv_ix[i]==true & model.tv_ix[j]==true
                k+=1
                A[j,i]=sdm_params[k]
                B[j,i]=sdm_params[k+o]
                if i==j
                    A[i,j]=VariableTransforms.from_R_to_pos(A[i,j])
                    B[i,j]=VariableTransforms.from_R_to_11(B[i,j])
                end
            end
        end
    end
    ω, A, B
end

function initialize_parameters!(model::DenseScoreFilter, result::ScoreFilterResults, ω::AbstractArray; A=0.01, B=0.99)
    A_tv=diagm(0=>fill(from_pos_to_R(A),model.no_tv))[:]
    B_tv=diagm(0=>fill(from_11_to_R(B),model.no_tv))[:]
    parameters!(result,[ω;A_tv;B_tv])
end

get_ωᶜ(model::DenseScoreFilter,ω,B)=(I-B)*ω
get_next_ft(model::DenseScoreFilter,i,ip1,ωᶜ,B,fts,A,Sti,∇t)=fts[:,ip1:ip1] .= ωᶜ.+B*view(fts,:,i).+A*(Sti\∇t)
