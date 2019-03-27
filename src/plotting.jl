using DataFrames: DataFrame
using Gadfly: plot, Geom, Theme, cm, draw, SVG, layer, Guide, Coord, vstack
using Printf: @sprintf

import Colors
get_Gadfly_colors(n)=Colors.distinguishable_colors(n, Colors.LCHab(70, 60, 240),
    transform=c -> Colors.deuteranopic(c, 0.5),
    lchoices=Float64[65, 70, 75, 80],
    cchoices=Float64[0, 50, 60, 70],
    hchoices=range(0,stop=330,length=24)
)
colors=map(x->convert(Colors.RGBA{Float64}, x), get_Gadfly_colors(10))
function lowlight_color(c,x) Colors.RGBA{Float64}(c.r, c.g, c.b, x) end
function plot_paths(x; title="", ix=-1, panels=[])
    if ix==-1
        ix=1:size(x[1][1])[1]
    end
    panel_colors=[]
    if length(panels)>0
        if length(panels[1])>1
            panel_colors=[i[2] for i in panels]
        else
            panel_colors=get_Gadfly_colors(length(x))
        end
    else
        panels=1:length(x)
        panel_colors=get_Gadfly_colors(length(x))
    end
    layers_k=1:size(x[1][1])[1]
    layers=Dict(ip=>[] for ip in layers_k)
    for (paths,labels,colour) in zip(x,panels,panel_colors)
        colour=lowlight_color(colour, 1.0)
        for ipath in layers_k
            if size(paths[1],1)<ipath
                continue
            end
            ddft=DataFrame(
            x=collect(1:size(paths[1])[2])[ix],
            estimated=paths[1][ipath,:][ix]
            )
            if length(paths)>1
                ddft[:low_bound]=paths[2][:,ipath][ix]
                ddft[:high_bound]=paths[3][:,ipath][ix]
                if length(paths)>3
                    ddft[:median]=paths[4][:,ipath][ix]
                    layers[ipath]=[layers[ipath]..., layer(ddft, x=:x,
                    y=:median, Geom.line,
                    Theme(default_color=lowlight_color(colour,0.5),line_width=4pt))
                    ]
                end
                layers[ipath]=[layers[ipath]..., layer(ddft, x=:x,
                    y=:estimated, Geom.line,
                    ymin=:low_bound, ymax=:high_bound, Geom.ribbon,
                    Theme(default_color=colour,
                    lowlight_color=x->lowlight_color(x,0.1)
                    ))
                ]
                layers[ipath]=[layers[ipath]..., layer(ddft, x=:x,
                    y=:low_bound, Geom.line,
                    Theme(default_color=lowlight_color(colour,0.5)))
                    ]
                layers[ipath]=[layers[ipath]..., layer(ddft, x=:x,
                    y=:high_bound, Geom.line,
                    Theme(default_color=lowlight_color(colour,0.5)))
                ]
            else
                layers[ipath]=[layers[ipath]..., layer(ddft, x=:x,
                y=:estimated, Geom.line,
                Theme(default_color=colour))
                ]
            end
        end
    end


    plots=[
        begin
            titlen=(title=="" || ip>1) ? "$ip" : title
            # display(panels)
            plot(layers[ip]...,
                Guide.xlabel(""), Guide.ylabel(""), Guide.title(titlen),
                Guide.manual_color_key("", map(x->string(x[1]),panels),map(x->Colors.RGB(x),panel_colors)),
                # Theme(key_position=:bottom),
                Coord.Cartesian(xmin=ix[1], xmax=ix[end])
            )
        end
        for ip in layers_k
    ];
    plots
end
function plot_paths_opt(sdm_params, title, sdm_criterion;
        model_name=:GAS,
        reference_paths=[],
        reference_labels=[]
    )
    res=sdm_criterion(sdm_params; return_paths=true)
    try
        x=@sprintf("%0.4e", res[:criterion_value])
        if title==""
            title="1; $(x)"
        else
            title="$title; $(x)"
        end
    catch
    end
    ps=plot_paths([
            [res[:ft]],
            [[i] for i in reference_paths]...
        ];
        ix=1:maximum(size(res[:ft])),
        # ix=Main.plot_ix,
        panels=[
            [model_name, colors[1]],
            [[l,colors[i+1]] for (i,l) in enumerate(reference_labels)]...
        ],
    title=title
    );
    draw(SVG(35cm, length(ps)*6cm), vstack(ps...))
end
function optim_plot_update!(res,plot_every,start_time,opt,plots,i,model_name,criterion,reference_paths,reference_labels)
    p = "x" in keys(res.metadata) ? res.metadata["x"] : res.metadata["centroid"]
    if plots==true && time()-start_time[1]>=plot_every
        Main.IJulia.clear_output(true)
        try
            plot_paths_opt(p, "Pass: $(i); estimating with $(opt); iter: $(res.iteration)", criterion;
                model_name=model_name,
                reference_paths=reference_paths,
                reference_labels=reference_labels,
            )
        catch
        end
        start_time[1]=time()
    end
    false
end
