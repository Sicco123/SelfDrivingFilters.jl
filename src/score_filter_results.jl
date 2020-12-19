mutable struct ScoreFilterResults{T<:Real}
    stage::Symbol
    parameters::Array{T,1}
    criterion_value::T
    paths::Array{T,2}
    init_St::Array{T,2}
    init_ft::Array{T,2}
    optimization_results::Any
    # optimization_results::Optim.MultivariateOptimizationResults
    ScoreFilterResults(S, stage)=begin
        x = new{S}()
        x.stage=stage
        x
    end
end
parameters(r::ScoreFilterResults)=r.parameters
criterion_value(r::ScoreFilterResults)=r.criterion_value
paths(r::ScoreFilterResults)=r.paths
init_St(r::ScoreFilterResults)=r.init_St
init_ft(r::ScoreFilterResults)=r.init_ft

parameters!(r::ScoreFilterResults,x::AbstractArray{T,1}) where {T<:Real}=begin r.parameters=x end
parameters!(r::ScoreFilterResults,x::AbstractArray{T,2}) where {T<:Real}=begin r.parameters=x[:] end
criterion_value!(r::ScoreFilterResults,x)=begin r.criterion_value=x end
paths!(r::ScoreFilterResults,x)=begin r.paths=x end
init_St!(r::ScoreFilterResults,x::AbstractArray{T,2}) where {T<:Real}=begin r.init_St=x end
init_ft!(r::ScoreFilterResults,x::AbstractArray{T,2}) where {T<:Real}=begin r.init_ft=x end
init_ft!(r::ScoreFilterResults,x::AbstractArray{T,1}) where {T<:Real}=begin r.init_ft=reshape(x,:,1) end
optim_res!(r::ScoreFilterResults,x)=r.optimization_results=x

parameters!(r::ScoreFilterResults,r_from::ScoreFilterResults)=parameters!(r,r_from.parameters)
criterion_value!(r::ScoreFilterResults,r_from::ScoreFilterResults)=criterion_value!(r,r_from.criterion_value)
paths!(r::ScoreFilterResults,r_from::ScoreFilterResults)=paths!(r,r_from.paths)
init_St!(r::ScoreFilterResults,r_from::ScoreFilterResults)=init_St!(r,r_from.init_St)
init_ft!(r::ScoreFilterResults,r_from::ScoreFilterResults)=init_ft!(r,r_from.init_ft)

mutable struct ScoreFilterResultsContainer{T<:Real}
    static::ScoreFilterResults{T}
    initial::ScoreFilterResults{T}
    best::ScoreFilterResults{T}
    ScoreFilterResultsContainer(S)=new{S}(
        ScoreFilterResults(S, :static),
        ScoreFilterResults(S, :initial),
        ScoreFilterResults(S, :best),
    )
end
