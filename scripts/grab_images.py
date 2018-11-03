import os
import urllib.request

dirname = os.path.dirname
IMAGES_BASE = dirname(dirname(os.path.realpath(__file__))) + "/images/"
URL_STUB = "http://www.pokestadium.com/sprites/black-white/"
SHINY = URL_STUB + "shiny/"
REGULAR_IMAGE_PATH = IMAGES_BASE + "regular/"
SHINY_IMAGE_PATH = IMAGES_BASE + "shiny/"

with open("pokemon_names.txt", "rb") as f:
    for cnt, name in enumerate(f):
        name = name.decode('utf-8').lower().strip('\n')
        print("Grabbing <{}> with URL <{}{}{}>".format(name, URL_STUB, name, ".png"))
        urllib.request.urlretrieve("{}{}{}".format(URL_STUB, name, ".png"), "{}{}{}".format(REGULAR_IMAGE_PATH, name, ".png"))
        print("Grabbing <{}> with URL <{}{}{}>".format(name, SHINY, name, ".png"))
        urllib.request.urlretrieve("{}{}{}".format(SHINY, name, ".png"), "{}shiny-{}{}".format(SHINY_IMAGE_PATH, name, ".png"))

