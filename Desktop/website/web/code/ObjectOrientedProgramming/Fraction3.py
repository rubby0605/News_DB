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
    
    # 取得分子 分母
    
    def get_num( self ) : 
        return self.num
    
    def get_den( self ) : 
        return self.den
    
    # 計算分數倍數
    def mul( self , m ) :
        return Fraction(self.num*m,self.den)

a = Fraction(3,5) # a 為 3/5
a.num = 4 # 等同 a._ _dict_ _[’num’] = 4
a.__dict__['den'] = 7 # 等同 a.den = 7
print(a) # 印出 4/7