struct CurrentSample
    data::Array
    T::Int
    N::Int
end
function add_data!(model::ScoreModel, data)
    model.data=CurrentSample(
        data,
        size(data,1),
        size(data,2)
    )
end
