from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
loadedModel = tf.keras.models.load_model('cnn.h5')

def array(x):
    img = Image.open(x)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    size = 28, 28
    img.thumbnail(size, Image.ANTIALIAS)

    enhancer = ImageEnhance.Brightness(img)
    factor = 0.5
    im = enhancer.enhance(factor)

    I = np.asarray(im.convert("L"))
    reshaped_img = I.reshape(1, 28, 28, 1)


    #im.convert("L").show()

    return reshaped_img







prediction = loadedModel.predict(array("plss.jpg"))

s = 'abcdefghiklmnopqrstuvwxy'

print(np.argmax(prediction))
print(s[np.argmax(prediction)])


