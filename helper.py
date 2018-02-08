import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def load_data(linux=False):

    if linux == True:
        training = pd.read_csv("/home/gianluca/progetti/facial_keypoint_detection/training.csv")
        test = pd.read_csv("/home/gianluca/progetti/facial_keypoint_detection/test.csv")
    else:
        training = pd.read_csv("/Users/gianlucatadori/Documents/Programming/kaggle_facial/training.csv")
        test = pd.read_csv("/Users/gianlucatadori/Documents/Programming/kaggle_facial/test.csv")

    return (training,test)


def plot_image(data,n,with_points=False,symmetric=False):

    x = data.iloc[n]
    image = [int(i) for i in x[30].split(" ")]
    image = np.asarray(image)
    image = image.reshape([96,96])

    if with_points == False:
        plt.imshow(image,cmap='gray')
    else:
        fig, ax = plt.subplots()
        ax.imshow(image,cmap='gray')
        for i in range(0,(len(x)-2),2):
            ax.plot(x[i],x[i+1],color="r",marker="+")
        plt.show()

        if symmetric == True:
            image = image

def plot_test(x,y,n):

    image = x[(n-1)]
    label = y[(n-1)] * 48 + 48
    image = image.reshape([96,96])
    fig,ax = plt.subplots()
    ax.imshow(image,cmap="gray")
    for i in range(0,(len(label)-1),2):
        ax.plot(label[i],label[i+1],color="r",marker="+")
    plt.show()


def preprocess_1d(data,test=False):

    df = data.dropna()

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    x = np.vstack(df['Image'].values) / 255
    x = x.astype(np.float32)


    if test == False:
        y = df[df.columns[:-1]].values
        y = (y-48)/48
        y  = y.astype(np.float32)
        x, y = shuffle(x, y, random_state=42)

        return (x,y)

    return (x)

def preprocess_2d(data,test=False):

    df = data.dropna()

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    x = np.vstack(df['Image'].values) / 255
    x = x.astype(np.float32)
    x = x.reshape(-1, 96, 96, 1)

    if test == False:
        y = df[df.columns[:-1]].values
        y = (y-48)/48
        y  = y.astype(np.float32)
        x, y = shuffle(x, y, random_state=42)

        return (x,y)

    return (x)
