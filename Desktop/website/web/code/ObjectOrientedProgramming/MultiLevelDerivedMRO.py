# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

class X: 
    pass
class Y: 
    pass
class Z: 
    pass

class A(X,Y): 
    pass
class B(Y,Z): 
    pass

class M(B,A,Z): 
    pass

print(M.mro())

# Output:
# [<class '__main__.M'>, <class '__main__.B'>,
# <class '__main__.A'>, <class '__main__.X'>,
# <class '__main__.Y'>, <class '__main__.Z'>,
# <class 'object'>]
