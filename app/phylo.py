import os
import numpy as np
import pandas as pd
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

WEIGHT = [-5.37720197, 2.15129582, -2.7828926, 12.3475064, -1.90558047]


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


"""Genetic Algorithms"""


def choose_from(options, n):
    idxs = np.random.choice(range(len(options)), n)
    return [options[i] for i in idxs]


def draw_normal(mu, sigma):
    return np.random.normal(mu, sigma, 1)[0]


def fitness(xv):
    qv = col_freq(xv)
    s = np.dot(WEIGHT, qv).sum()
    return np.exp(s) / (1.0 + np.exp(s))


def crossover(p1, p2, x, y, r):
    m1 = np.array(p1).reshape(M, M)
    m2 = np.array(p2).reshape(M, M)
    for yi in range(y - r, y + r + 1):
        for xi in range(x - r, x + r + 1):
            if in_im(xi, yi):
                from1 = np.uint8(m1[yi][xi])
                from2 = np.uint8(m2[yi][xi])
                if from2 > 0:
                    m1[yi][xi] = from2
                if from1 > 0:
                    m2[yi][xi] = from1
    c1 = m1.reshape(M**2)
    c2 = m2.reshape(M**2)
    return c1, c2


def mutate_form(cx, px):
    subsample = np.array([cx, cx, px, px] + [generate_random() for i in range(2)])
    ev = expect_from_subsample(subsample)
    bv = outline_body(ev)
    return bv


def rank_generation(generation):
    rdf = pd.DataFrame()
    rdf["x"] = generation
    rdf["fitness"] = [fitness(xv) for xv in generation]
    rdf = rdf.sort_values(by="fitness", ascending=False).reset_index(drop=True)
    return rdf


def imrow(vecs, scale=1):
    ims = [showim(v, scale) for v in vecs]
    full = Image.new(MODE, (len(ims) * M * scale, M * scale))
    for i in range(len(ims)):
        full.paste(ims[i], (i * M * scale, 0))
    return full


def imgrid(rows, scale=1):
    ims = [imrow(row, scale) for row in rows]
    full = Image.new(MODE, (len(rows[0]) * M * scale, len(rows) * M * scale))
    for i in range(len(ims)):
        full.paste(ims[i], (0, i * M * scale))
    return full


def neighbors_block(x, y, r=1):
    neigh = []
    for xi in range(x - r, x + r + 1):
        for yi in range(y - r, y + r + 1):
            if in_im(xi, yi):
                neigh.append((xi, yi))
    return neigh


def smoother(method=None, threshold=0.25, radius=1):
    def do_transform(cell, x, y, mat):
        neigh = neighbors_block(x, y, radius)
        n_vals = [mat[yn][xn] for xn, yn in neigh]
        palette = []
        if method == "light":
            palette = range(1, Q)
        if method == "dark":
            palette = range(Q - 1, 0, -1)
        if method == "mid":
            palette = [2, 1, 3]
        for target in palette:
            match = [1 if v == target else 0 for v in n_vals]
            ratio = sum(match) / len(match)
            if ratio >= threshold:
                return target
        return cell
    return do_transform


def smooth_quanta(xv, method, threshold=0.25, radius=1):
    sm = smoother(method, threshold, radius)
    cv = cell_transform(xv, sm)
    bv = outline_body(cv)
    return bv
