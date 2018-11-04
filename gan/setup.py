from setuptools import setup, find_packages

setup(name="phylo",
        version="0.0",
        packages=find_packages(),
        description="GAN for Phylo",
        author="Meme Dream Team",
        author_email="rchatrath7@gmail.com",
        license="MIT",
        install_requires=[
            'keras',
            'matplotlib',
            'numpy',
            'seaborn',
            'imageio',
        ],
        zip_safe=False)

