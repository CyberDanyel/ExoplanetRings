# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:07:54 2023

@author: victo
"""

# testing for the ring phase curve

import raytracer

rays = [raytracer.Ray(pos = [i%100, int(i/100), 0]) for i in range(100)]

