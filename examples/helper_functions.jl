
function structural_breaks(T, levels, proportions)
    proportions=1.0.*proportions./sum(proportions)
    sizes=round.(Int,T.*proportions)
    sizes[end]+=T-sum(sizes)
    vcat([
        fill(levels[i],sizes[i])
        for i=1:length(levels)
    ]...)
end
function sinusoid(T, amplitude, cycles)
    (amplitude*sin.(0:cycles*2π/T:cycles*2π))[1:T]
end

function plot_models(true_paths, models)
    ps=[
        plot(
            vcat(
                DataFrame(x=1:length(true_paths[i,:]), y=true_paths[i,:], t=:true),
                [DataFrame(x=1:length(models[k].paths[i,:]), y=models[k].paths[i,:], t=k) for k in keys(models)]...
            ),
            x=:x, y=:y, color=:t,
            Geom.line,
            Coord.Cartesian(xmin=1,xmax=T+1),
            Guide.xlabel(""), Guide.ylabel("parameter $i")
        )
        for i=1:length(tv_ix)
    ]
    draw(SVGJS(30cm,10cm*length(ps)),vstack(ps...))
end
function plot_models(models)
    ps=[
        plot(
            vcat(
                [DataFrame(x=1:length(models[k].paths[i,:]), y=models[k].paths[i,:], t=k) for k in keys(models)]...
            ),
            x=:x, y=:y, color=:t,
            Geom.line,
            Coord.Cartesian(xmin=1,xmax=length(models[first(keys(models))].paths[1,:])),
            Guide.xlabel(""), Guide.ylabel("parameter $i")
        )
        for i=1:length(tv_ix)
    ]
    draw(SVGJS(30cm,10cm*length(ps)),vstack(ps...))
end