#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 06:45:50 2022

@author: johnlangham
"""
#
# A_AI_matrix_loss_01_vnn.py
#
# This is stored as py so it is not explicitly in the .ipynb, or could be
# delivered as a .pyc
#
# v01 - 1st cut
#
def check_loss(Y, F, L=None):
    #
    # Do rounding and do try
    #
    import numpy as np
    #
    OK = False
    #
    try:
        L   = np.round(L, 4)
        myL = np.round(-(Y * np.log(F) + (1 - Y) * np.log(1 - F)), 4)
        #
        if np.array_equal(L, myL):
            OK = True
        #
    except:
        print("Sorry - there is some problem with your data,",
              "please check and try again")
        return
    #
    if OK:
        print("Correct - Well done!")
    else:
        print("Sorry - your answer isn't quite right. Have another go")
    #
#
#