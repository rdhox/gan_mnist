#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:44:10 2021

@author: renaud
"""
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
from numpy import asarray

def create_latent_points(latent_dim, n):
    # create vector within range of value
    input_x = randn(latent_dim*n)
    input_x = input_x.reshape((n, latent_dim))
    return input_x

def save_plot(ex, n=5):
    for i in range(n*n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis('off')
        pyplot.imshow(ex[i, :, :, 0], cmap='gray_r')
    pyplot.show()
    
model = load_model('bestmodels/generator_model_050.h5')
# generate a bunch of random number image
# latent_points = create_latent_points(100, 25)
# generate_samples = model.predict(latent_points)
# save_plot(generate_samples)

# generate one image
# vector = asarray([[1.0 for _ in range(100)]])

vector = asarray([[
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0,
]])
generate_im = model.predict(vector)
save_plot(generate_im, 1)