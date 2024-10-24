struct CurrentSample{T<:Real}
    data::Array{T,2}
    T::Int
    N::Int
end
function add_data!(model::ScoreFilter, data::Array{T,2}; time_axis=1) where {T<:Real}
    if time_axis==1
        data=Matrix(data')
    end
    model.data=CurrentSample(
        data,
        size(data,2),
        size(data,1)
    )
    model.results=ScoreFilterResultsContainer(eltype(data))
    model.storage=ScoreFilterStorage(eltype(data), model)
    model.data
end
add_data!(model::ScoreFilter, data::Array{T,1}) where {T<:Real} = add_data!(model, reshape(data,1,:); time_axis=2)

function update_data!(model::ScoreFilter, data::Array{T,2}; time_axis=1) where {T<:Real}
    if time_axis==1
        data=Matrix(data')
    end
    model.data = CurrentSample(
        data,
        size(data,2),
        size(data,1)
    )
    model.data
end

update_data!(model::ScoreFilter, data::Array{T,1}) where {T<:Real} = update_data!(model, reshape(data,1,:); time_axis=2)