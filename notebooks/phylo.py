import os
import numpy as np
import math
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
BACKGROUND = (238, 255, 188, 255)

image_path_stub = "/".join(os.getcwd().split("/")[0:-1])
REGULAR_POKEMON_PATH = image_path_stub + "/images/regular/"
SHINY_POKEMON_PATH = image_path_stub + "/images/shiny/"


"""Converting Between Vectors and Images"""


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


"""Loading Images"""

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


"""Computing Expected Vectors"""


def col_counts(col):
    buckets = np.zeros(Q + 1)
    for x in col:
        buckets[x] += 1
    return buckets


def col_freq(col):
    buckets = col_counts(col)
    freq = buckets / buckets.sum()
    return freq


def col_exp(col):
    freq = col_freq(col)
    expt = np.dot(freq, np.array(range(Q + 1))).sum()
    return expt


def draw_val(col):
    expt = col_exp(col)
    return math.floor(expt)


def generate_expected(train_pop, n_sub, seed=None):
    if seed:
        np.random.seed(seed)
    idxs = list(np.random.choice(range(len(train_pop)), n_sub))
    subsample = np.array([train_pop[i] for i in idxs])
    ev = np.array([draw_val(subsample[:, j]) for j in range(M**2)])
    return ev


def expect_from_subsample(subsample):
    ev = np.array([draw_val(subsample[:, j]) for j in range(M**2)])
    return ev


def active_prop(xv):
    max_active = M**2
    active = sum(map(lambda x: 1 if x > 1 else 0, xv))
    return active / max_active


def in_im(x, y):
    return x >= 0 and x < M and y >= 0 and y < M


def neighbors4(x, y, r=1):
    neigh = [(x - r, y), (x + r, y), (x, y - r), (x, y + r)]
    neigh = list(filter(lambda t: in_im(*t), neigh))
    return neigh


def cell_transform(xv, do_transform):
    mat = xv.reshape(M, M)
    out = np.uint8(np.zeros(M**2).reshape(M, M))
    for y, row in enumerate(mat):
        for x, cell in enumerate(row):
            out[y][x] = do_transform(cell, x, y, mat)
    return out.reshape(M**2)


def clean_fluff(xv, radius=4):
    def do_transform(cell, x, y, mat):
        neigh = neighbors4(x, y, radius)
        n_vals = [mat[yn][xn] for xn, yn in neigh]
        if max(n_vals) <= 1:
            return 0
        return cell
    return cell_transform(xv, do_transform)


def outline_body(xv, radius=4):
    def do_transform(cell, x, y, mat):
        neigh = neighbors4(x, y, radius)
        n_vals = [mat[yn][xn] for xn, yn in neigh]
        if max(n_vals) <= 1:
            return 0
        if max(n_vals) == 2 and cell <= 1 and sum(n_vals) < 6:
            return Q
        return cell
    return cell_transform(xv, do_transform)


def show_trans(im):
    pixels = np.array(im).reshape(M * M, C)
    new_pix = [BACKGROUND if p[-1] == TRANSPARENT else p for p in pixels]
    new_pix = np.array(np.uint8(new_pix)).reshape(M, M, C)
    return Image.fromarray(new_pix, mode=MODE)


def showim(xv, scale=1, outline=False):
    vec = xv
    if outline:
        vec = outline_body(xv)
    return scale_img(show_trans(unvectorize(vec)), scale)
