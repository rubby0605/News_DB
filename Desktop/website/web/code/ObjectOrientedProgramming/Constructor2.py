# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:29:47 2021

@author: Alvin
"""

class ComplexNumber:

    def __init__(self, r = 0, i = 0):
        """"初始化方法"""
        self.real = r 
        self.imag = i 

    def getData(self):
        print("{0}+{1}j".format(self.real, self.imag))

if __name__ == '__main__':
    c = ComplexNumber(5, 6)
    c.getData()

    c2 = ComplexNumber(10, 20)
    
    # 試著賦值給一個未定義的屬性
    c2.attr = 120
    print("c2 = > ", c2.attr)

    print("c.attr => ", c.attr)
    


