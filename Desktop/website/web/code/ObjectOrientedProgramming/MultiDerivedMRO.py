# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

class Base1:
    pass

class Base2:
    pass

class MultiDerived(Base1, Base2):
    pass
    
print(MultiDerived.__mro__)
print(MultiDerived.mro())

