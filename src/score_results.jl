mutable struct ScoreResults
    stage::Symbol
    parameters::Array
    criterion_value::Float64
    paths::Array
    init_St::Array
    init_ft::Array
end
parameters(r::ScoreResults)=r.parameters
criterion_value(r::ScoreResults)=r.criterion_value
paths(r::ScoreResults)=r.paths
init_St(r::ScoreResults)=r.init_St
init_ft(r::ScoreResults)=r.init_ft

parameters!(r::ScoreResults,x)=begin r.parameters=x end
criterion_value!(r::ScoreResults,x)=begin r.criterion_value=x end
paths!(r::ScoreResults,x)=begin r.paths=x end
init_St!(r::ScoreResults,x::Array)=begin r.init_St=x end
init_ft!(r::ScoreResults,x::Array)=begin r.init_ft=x end

parameters!(r::ScoreResults,r_from::ScoreResults)=parameters!(r,r_from.parameters)
criterion_value!(r::ScoreResults,r_from::ScoreResults)=criterion_value!(r,r_from.criterion_value)
paths!(r::ScoreResults,r_from::ScoreResults)=paths!(r,r_from.paths)
init_St!(r::ScoreResults,r_from::ScoreResults)=init_St!(r,r_from.init_St)
init_ft!(r::ScoreResults,r_from::ScoreResults)=init_ft!(r,r_from.init_ft)

mutable struct ScoreResultsContainer
    static::ScoreResults
    initial::ScoreResults
    best::ScoreResults
    final::ScoreResults
end
