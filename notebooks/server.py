"""Setting Up Flask"""

import numpy as np
import pandas as pd
import phylo as ph
import scipy.spatial.distance as scidist

from flask import Flask
# from flask import render_template
from flask import jsonify
from flask import request
from flask import send_from_directory
from flask_cors import CORS
server = Flask(__name__)
CORS(server)


"""Load Creatures"""


def binarize(xv):
    return np.array([ph.Q if v > 0 else 0 for v in xv])


np.random.seed(820)
REGULAR_POKEMON, FILE_PATHS = np.load("../notebooks/reg.pickle")

binary_pokemon = [binarize(xv) for xv in REGULAR_POKEMON]
poke_df = pd.DataFrame()
poke_df["i"] = range(len(binary_pokemon))
poke_df["x"] = binary_pokemon
print("Ready to search!")

"""Serving HTML Pages/Templates"""


@server.route("/bulma.min.css")
def style():
    return send_from_directory("templates", "bulma.min.css")


@server.route("/sketchsearch")
def sketchsearch():
    try:
        text = request.args.get("vector")
        skv = [int(t) for t in text.split(",")]
        poke_df["hamming"] = [scidist.hamming(skv, xv) for xv in binary_pokemon]
        top = poke_df.sort_values(by="hamming", ascending=True).head(5).reset_index(drop=True)
        idx = top["i"][0]
        url0 = FILE_PATHS[top["i"][0]]
        url1 = FILE_PATHS[top["i"][1]]
        url2 = FILE_PATHS[top["i"][2]]
        return jsonify({"sum": sum(skv), "url0": url0, "url1": url1, "url2": url2, "idx": int(idx)})
    except Exception as e:
        return jsonify({"error": str(e)})
