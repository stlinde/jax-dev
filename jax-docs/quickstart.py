# jax-docs/quickstart.py
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import time

"""Multiplying matrices"""
key = random.PRNGKey(0)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)

@jit
def mult_mat(x):
    start = time.time()
    jnp.dot(x, x.T)
    end = time.time()
    return end - start
    
