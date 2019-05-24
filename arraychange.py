#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:17:12 2019

@author: yang
"""

import numpy as np

box = np.array([[1,2],[3,4],[5,6],[7,8]])
print(box)
#box[1],box[3] = box[3],box[1]
#box[3],box[1] = box[1],box[3]
box[1, 0],box[3, 0] = box[3, 0],box[1, 0]
                           
box[1, 1],box[3, 1] = box[3, 1], box[1, 1]
print('after change:{}'.format(box))