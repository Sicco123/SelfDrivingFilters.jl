mutable struct ScoreFilterStorage{T<:Real}
    criterion_values::Array{T,1}
    paths::Array{T,2}
    ScoreFilterStorage(S,model)=begin
        x = new{S}()
        x.paths=Array{S}(undef, model.data.N, model.data.T)
        x.criterion_values=Array{S}(undef, model.data.T)
        x
    end
end