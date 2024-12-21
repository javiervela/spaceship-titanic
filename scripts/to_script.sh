#!/bin/bash

poetry run jupyter nbconvert --to script spaceship-titanic.ipynb && sed -i 's/^.*plt\.show.*$/# &/; s/display(/print(/g' spaceship-titanic.py
