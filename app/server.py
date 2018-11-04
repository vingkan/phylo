"""Setting Up Flask"""

import numpy as np
import pandas as pd
import phylo as ph
import os
import scipy.spatial.distance as scidist

from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
server = Flask(__name__)


"""Load Creatures"""


def binarize(xv):
    return np.array([ph.Q if v > 0 else 0 for v in xv])


paths = []
for (dirnames, dirpath, filenames) in os.walk(ph.REGULAR_POKEMON_PATH):
    paths.extend(filenames)

np.random.seed(820)
REGULAR_POKEMON = ph.vectorize_pokemon(ph.REGULAR_POKEMON_PATH)
SHINY_POKEMON = ph.vectorize_pokemon(ph.SHINY_POKEMON_PATH)

binary_pokemon = [binarize(xv) for xv in REGULAR_POKEMON]
poke_df = pd.DataFrame()
poke_df["i"] = range(len(binary_pokemon))
poke_df["x"] = binary_pokemon


@server.route('/hello')
def hello():
    return 'Hello World!'


"""Serving HTML Pages/Templates"""


@server.route('/')
def home():
    return render_template('index.html')


@server.route('/sketchsearch')
def sketchsearch():
    text = request.args.get("vector")
    skv = [int(t) for t in text.split(",")]
    poke_df["hamming"] = [scidist.hamming(skv, xv) for xv in binary_pokemon]
    top = poke_df.sort_values(by="hamming", ascending=True).head(5).reset_index(drop=True)
    url = paths[top["i"][0]]
    return jsonify({"sum": sum(skv), "url": url})
