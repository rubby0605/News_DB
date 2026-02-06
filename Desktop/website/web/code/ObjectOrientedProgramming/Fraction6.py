# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 計算兩數的 gcd
def gcd( a , b ) :
    a , b = abs(a) , abs(b)
    if a > b :
        return gcd(a%b,b) if a%b else b
    else :
        return gcd(b%a,a) if b%a else a

class Fraction :
    def __init__ ( self , n = 0 , d = 1 ) :
        self.num , self.den = n , d
        
    #字串表示方法：設定物件的「字串」輸出樣式
    def __str__ ( self ) :
        if self.den == 1 :
            return str(self.num)
        else :
            return str(self.num) + '/' + str(self.den)
    # 計算最簡分數
    def simplest_form( self ) :
        g = gcd(self.num,self.den)
        return Fraction(self.num//g,self.den//g)

a = Fraction(16,28)
# 印出：4/7
print( a.simplest_form() )
