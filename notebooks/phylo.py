import os
import numpy as np
import imageio
from PIL import Image

Q = 4
M = 96
C = 4
MODE = "RGBA"
LUM_WEIGHT = [0.21, 0.72, 0.07]
LIGHTEST = 255
TRANSPARENT = 0
OPAQUE = 255
NO_COLOR = 0

image_path_stub = "/".join(os.getcwd().split("/")[0:-1])
REGULAR_POKEMON_PATH = image_path_stub + "/images/regular/"
SHINY_POKEMON_PATH = image_path_stub + "/images/shiny/"


def quantize(val, alpha):
    if alpha == TRANSPARENT:
        return 0
    elif val >= 192:
        return 1
    elif val >= 128:
        return 2
    elif val >= 64:
        return 3
    else:
        return 4


def unquantize(val):
    alpha = TRANSPARENT if val == NO_COLOR else OPAQUE
    color_markers = [0, 255, 160, 96, 0]
    pixel_val = color_markers[val]
    return pixel_val, alpha


def vectorize(im):
    pixels = np.array(im).reshape(M * M, C)
    vec = np.zeros(M * M)
    for i in range(len(pixels)):
        px = pixels[i]
        alpha = px[-1]
        single_val = np.dot(LUM_WEIGHT, px[:-1]).sum()
        vi = quantize(single_val, alpha)
        vec[i] = vi
    return np.uint8(vec)


def unvectorize(vec):
    comps = [unquantize(v) for v in vec]
    imarr = [[v, v, v, a] for v, a in comps]
    outarr = np.array(np.uint8(imarr)).reshape(M, M, C)
    im = Image.fromarray(outarr, mode=MODE)
    return im


def load_img(filename):
    im = imageio.imread(filename, pilmode=MODE)
    im = Image.fromarray(im)
    return im


def load_all_images(path):
    paths = []
    for (dirnames, dirpath, filenames) in os.walk(path):
        paths.extend(filenames)

    return [load_img(path + filename) for filename in paths]


def scale_img(img, k):
    return img.resize((k * img.size[0], k * img.size[0]))


def generate_random(seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.randint(Q + 1, size=M**2)


def vectorize_pokemon(path):
    print("Vectorizing pokemon from {}".format(path))
    images = load_all_images(path)
    pokemon = []

    for i in range(len(images)):
        if i % 100 == 0:
            print("Vector iteration {}".format(i))

        pokemon.append(vectorize(images[i]))

    print("Done vectorizing")
    return np.asarray(pokemon)

def save_image(img, filepath):
    result = unvectorize(img)
    result.save("output/"+filepath)
