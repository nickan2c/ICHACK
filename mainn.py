from cameratest import Camera
from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np

def array(x):
    img = Image.open(x)
    #img = img.transpose(Image.FLIP_LEFT_RIGHT)

    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    size = 28, 28
    img.thumbnail(size, Image.ANTIALIAS)

    enhancer = ImageEnhance.Brightness(img)
    factor = 0.8
    im = enhancer.enhance(factor)

    I = np.asarray(im.convert("L"))
    I = I/255
    print(I)
    reshaped_img = I.reshape(1, 28, 28, 1)


    #im.convert("L").show()

    return reshaped_img

def predict(x,model):
    # input for convolutional neural network is shaped like this

    prediction = model.predict(array(x))

    s = 'abcdefghijklmnopqrstuvwxyz'
    num = np.argmax(prediction)
    print(prediction)
    print(len(prediction[0]))
    print(num)
    print(s[num])
    return s[num]


letters = []
num_imgs = Camera.main()-1
loadedModel = tf.keras.models.load_model('cnn.h5')

for i in range(num_imgs):
    filename = f'pic{i+1}.jpg'
    letters.append(predict(filename,loadedModel))

print(''.join(letters))









