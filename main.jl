using Knet, Images

function get_training_data()
    X = []
    Y = []
    bad = 0
    good = 0
    for filename in readdir("thumb")[1:20]
        name, rating = split(replace(filename, ".thumbnail.jpg", ""),"@")
        rating = parse(Int16, rating)
        rating < 67 ? push!(Y,0) : push!(Y,1)
        raw_img = load("thumb/$(filename)")
        img = permutedims(channelview(raw_img), (3,2,1))
        img = convert(Array{Float32}, img)
        img = reshape(img[:,:,1:3], (150,101,3,1))
        push!(X, permutedims(img, [2,1,3,4]))
    end
    cat(4,X...), Y
end


function predict(w,x)
    n=length(w)-4
    for i=1:2:n
        x = pool(relu.(conv4(w[i],x;padding=0) .+ w[i+1]))
    end
    x = mat(x)
    for i=n+1:2:length(w)-2
        x = relu.(w[i]*x .+ w[i+1])
    end
    return w[end-1]*x .+ w[end]
end

loss(w,x,ygold) = nll(predict(w,x), ygold)

lossgradient = grad(loss)

function train(w, data)
    for (x,y) in data
        g = lossgradient(w, x, y)
        update!(w, g, lr=0.12)
    end
    return w
end

function weights()
    w = Array{Any}(8)
    w[1] = xavier(5,5,1,20)
    w[2] = zeros(1,1,20,1)
    w[3] = xavier(5,5,20,50)
    w[4] = zeros(1,1,50,1)
    w[5] = xavier(500,800)
    w[6] = zeros(500,1)
    w[7] = xavier(10,500)
    w[8] = zeros(10,1)
    return map(a->convert(Array,a), w)
end


function main()
    X, Y = get_training_data()
    global dtrn = minibatch(X, Y, 5; xtype=Array)
    global dtst = minibatch(X, Y, 5; xtype=Array)

    w = weights()
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))

    report(0)
    for epoch=1:5
        train(w, dtrn)
        report(epoch)
    end
end
