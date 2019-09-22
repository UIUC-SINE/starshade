from setuptools import setup

setup(
    name='uiuc-starshade',
    version='1',
    packages=['starshade'],
    author="Ulas Kamaci, Jamila Taaki",
    author_email="kamaci.ulas@gmail.com",
    description="Starshade Imaging Simulations",
    long_description=open('README.md').read(),
    license="GPLv3",
    keywords="starshade uiuc",
    url="https://github.com/uiuc-sine/starshade",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy"
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
