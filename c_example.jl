for p in ("Knet", )
    (Pkg.installed(p) == nothing) && Pkg.add(p)
end

using Knet, Images
include(Pkg.dir("Knet", "data", "cifar.jl"))


function get_training_data()
    X = []
    Y = Array{UInt8,1}()
    bad = 0
    good = 0
    for filename in readdir("thumb")[1:1536]
        name, rating = split(replace(filename, ".thumbnail.jpg", ""),"@")
        rating = parse(Int16, rating)
        rating < 67 ? push!(Y,0) : push!(Y,1)
        raw_img = load("thumb/$(filename)")
        img = permutedims(channelview(raw_img), (3,2,1))
        img = convert(Array{Float32}, img)
        img = reshape(img[:,:,1:3], (32,32,3,1))
        push!(X, permutedims(img, [2,1,3,4]))
    end
    cat(4,X...), Y, cat(4,X...), Y
end




function loaddata()
    Xtrn, Ytrn, Xtest, Ytest = get_training_data()

    #= Subtract mean of each feature
    where each channel is considered as
    a single feature following the CNN
    convention=#

    mn = mean(Xtrn, (1,2,4))
    Xtrn = Xtrn .- mn
    Xtest = Xtest .- mn
    return (Xtrn, Ytrn), (Xtest, Ytest)
end

# The global device setting (to reduce gpu() calls)
let at = nothing
    global atype
    atype() = (at == nothing) ? (at = (gpu() >= 0 ? KnetArray: Array)) : at
end


##Model definition
kaiming(et, h, w, i, o) = et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)

function init_model(;et=Float32)
    w = Any[
        kaiming(et, 3, 3, 3, 16),    bnparams(et, 16),
        kaiming(et, 3, 3, 16, 32),   bnparams(et, 32),
        kaiming(et, 3, 3, 32, 64),   bnparams(et, 64),
        xavier(et, 100, 8 * 8 * 64), bnparams(et, 100),
        xavier(et, 2, 100),         zeros(et, 2, 1)
    ]
    # Initialize a moments object for each batchnorm
    m = Any[bnmoments() for i = 1:4]
    w = map(atype(), w)
    return w, m
end

function conv_layer(w, m, x; maxpool=true)
    o = conv4(w[1], x; padding=1)
    o = batchnorm(o, m, w[2])
    o = relu.(o)
    if maxpool; o=pool(o); end
    return o
end

function lin_layer(w, m, x)
    o = w[1] * x
    o = batchnorm(o, m, w[2])
    return relu.(o)
end

function predict(w, m, x)
    o = conv_layer(w[1:2] , m[1], x)
    o = conv_layer(w[3:4] , m[2], o)
    o = conv_layer(w[5:6] , m[3], o; maxpool=false)
    o = lin_layer( w[7:8] , m[4], mat(o))
    return w[9] * o .+ w[10]
end

function loss(w, m, x, classes)
    ypred = predict(w,m, x)
    println(ypred, classes)
    loss = nll(ypred, classes)
    println(loss)
    return loss
end

lossgrad = grad(loss)

# Training
function epoch!(w, m, o, xtrn, ytrn;  mbatch=64)
    data = minibatch(xtrn, ytrn, mbatch;
                   shuffle=true,
                   xtype=atype())
    for (x, y) in data
        println("going through batch !!")
        g = lossgrad(w, m, x, y)
        update!(w, g, o)
    end
end

# Accuracy computation
function acc(w, m, xtst, ytst; mbatch=64)
    data = minibatch(xtst, ytst, mbatch;
                     partial=true,
                     xtype=atype())
    model = (w, m)
    return accuracy(model, data,
                    (model, x)->predict(model[1], model[2], x);
                    average=true)
end

function train()
    w, m = init_model()
     o = map(_->Momentum(;lr=0.2), w)
    (xtrn, ytrn), (xtst, ytst) = loaddata()
    for epoch = 1:5
        println("epoch: ", epoch)
        epoch!(w, m, o, xtrn, ytrn)
        println("train accuracy: ", acc(w, m, xtrn, ytrn))
        println("test accuracy: ", acc(w, m, xtst, ytst))
    end
end
