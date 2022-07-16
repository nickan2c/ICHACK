from PIL import Image
import numpy
import PIL
# load the image
image =('A1.jpg')
# summarize some details about the image
I = numpy.asarray(Image.open(image).convert("L"))
I = I.reshape(40000)
print(I.shape)


def convertArray(img):
    I = numpy.asarray(Image.open(image).convert("L"))
    return I

# ytrain i is A,B...
yTrain[i] = myDict[yTrain[i]]
