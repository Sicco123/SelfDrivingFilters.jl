module VariableTransforms
    from_11_to_R(x)=log(1+x)-log(1-x)
    from_R_to_11(x)=2*exp(x)/(1+exp(x))-1.
    from_01_to_R(x)=log(x)-log(1-x)
    from_R_to_01(x)=exp(x)/(1+exp(x))
    from_pos_to_R(x)=log(x)
    from_R_to_pos(x)=exp(x)
end # module
