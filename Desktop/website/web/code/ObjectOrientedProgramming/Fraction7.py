# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
class Fraction :
    def __init__ ( self , n = 0 , d = 1 ) :
        self.num , self.den = n , d
        
    #字串表示方法：設定物件的「字串」輸出樣式
    def __str__ ( self ) :
        if self.den == 1 :
            return str(self.num)
        else :
            return str(self.num) + '/' + str(self.den)
        
    # 設定分子與分母
    def set_val( self , n , d = 1 ) :
        self.num , self.den = n , d
    
a = b = Fraction(3,4) # a 與 b 為同一個物件
b.set_val(5,6) # a 與 b 都是 5/6
print( a is b ) # True
a = Fraction(1,2) # a 為 1/2 , b = 5/6，a 為新物件
print( a is b ) # False

