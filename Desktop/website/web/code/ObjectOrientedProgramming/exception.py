# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 20:35:34 2021

@author: teacher
"""

class B(BaseException):
    pass
    #def __str__(self):
    #    return "hahahaha"

class C(B):
    pass

class D(C):
    pass

for c in [B,C,D]:
    try:
        raise c()
    except D:
        print("D")
    except C:
        print("C")        
    except B:
        print("B")        