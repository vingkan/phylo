# Phylo

We love creature sprites like this image of Ivysaur:

![Ivysaur Sprite](http://www.pokestadium.com/sprites/black-white/ivysaur.png)

Sprites show surprising levels of detail despite their small image sizes. However, under every beautiful sprite is a matrix of numbers. Sprite of this size, whether used in games, websites, or other forms of media, can be expressed with 96 x 96 x 4 x 255 values.

Since every sprite has a unique mathematical fingerprint, we wondered if we could discover new creatures in the vector space of such images. We were excited to get creative in the face of a daunting mathematical challenge.

We applied a one-way transformation to all of the sprites, reducing them to 96 x 96 = 9216 component vectors. With five color values for each component, this still left a search space of over five hundred quadrillion (17 zeros) possible vectors. Far too many for humans to look at individually.

We applied a slew of data mining, machine learning, artificial intelligence, and deep learning techniques in pursuit of new, never-before-seen creature sprites. Methods implemented include:

- Principle Components Analysis
- K-Means Clustering
- Logistic Regression
- Kernel Smoothing
- Genetic Algorithms
- Linear Optimization
- Generative Adversarial Neural Networks
- Recommender Systems

In our search, we uncovered many, many, many pixelated red herrings as well as a handful of new creature-like sprites. They are "rough around the edges" but born entirely out of mathematics and computer science.

We developed [Jupyter notebooks](https://github.com/vingkan/phylo/blob/master/notebooks/Color%20Frequency%20Classifier.ipynb) and a [website](https://vingkan.github.io/phylo/) to share our findings. The results show intriguing clusters of creatures and curious mutations of prospective sprites.

[View methods and results on our website.](https://vingkan.github.io/phylo/)

We also applied our sprite/vector transformation process to create a tool that allows users to search for sprites by drawing them. _We have turned off the server that supports this, so the draw to search feature on the website will not work._
