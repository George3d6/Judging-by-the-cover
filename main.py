import os, sys
import numpy as np
from PIL import Image
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score

def thumbnelify_dataset():
    def thumbneilify(dir, file_name):
        size = 32, 32
        thumb_file = "thumb/" + file_name[:-4]
        outfile =  thumb_file + ".thumbnail.jpg"
        if thumb_file != outfile:
            try:
                im = Image.open(dir + file_name)
                thumb = im.resize(size)
                thumb.save(outfile, "JPEG")
            except IOError:
                print("cannot create thumbnail for '%s'" % thumb_file)

    for subdir, dirs, files in os.walk("original"):
        for file in files:
            thumbneilify("original/", file)

def append_ref(arr, ele):
    arr = np.append(arr, ele)

def to_numpy_array(image):
    r, g ,b = image.split()

    r_vec = np.asanyarray(r)

    g_vec = np.asanyarray(g)

    b_vec = np.asanyarray(b)

    return np.append(np.append(r_vec, g_vec), b_vec)

thumbnelify_dataset()
exit()

X = []
Y = []

for subdir, dirs, files in os.walk("thumb"):
    for file in files:
        try:
            arr = to_numpy_array(Image.open("thumb/" + file))
            X.append(arr)
            score = round(int(file[file.index('@') + 1:][:2].replace(".", ""))/10)
            Y.append(score)
            print("Appedning movie: {}, with score: {}".format(file , score))
        except Exception as e:
            print("Error appending to train&test set: {}".format(e))

print(len(X[3]))

model = MLPClassifier(solver='adam', hidden_layer_sizes=(int(2000),int(2000/2),int(2000/4),int(2000/8),
                                                         int(2000/16),
                                                        int(2000/8),int(2000/4),int(2000/2),int(2000)),
                      random_state=1, learning_rate="adaptive", shuffle=True,
                      max_iter=400, batch_size = 260)

X_train = X[0:1240]
Y_train = Y[0:1240]

X_test = X[1240:]
Y_test = Y[1240:]

model.fit(X_train, Y_train)
predictions = model.predict(X_test)

for i in range(0, len(predictions)):
    print("Predicted: {} , Real: {}, Difference: {}".format(predictions[i], Y_test[i], predictions[i] - Y_test[i]))

predictions_rounded = []
for p in predictions:
    predictions_rounded.append(round(p))

print("Accuracy: {}", accuracy_score(Y_test, predictions))
print("Perfect accuracy can be: {}", accuracy_score(Y_test, Y_test))
