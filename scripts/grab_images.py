import os
import urllib.request

dirname = os.path.dirname
images_path = dirname(dirname(os.path.realpath(__file__))) + "/images/"
URL_STUB = "http://www.pokestadium.com/sprites/black-white/"

with open("pokemon_names.txt", "rb") as f:
    for cnt, name in enumerate(f):
        name = name.decode('utf-8').lower().strip('\n')
        print("Grabbing <{}> with URL <{}{}{}>".format(name, URL_STUB, name, ".png"))
        urllib.request.urlretrieve("{}{}{}".format(URL_STUB, name, ".png"), "{}{}{}".format(images_path, name, ".png"))

