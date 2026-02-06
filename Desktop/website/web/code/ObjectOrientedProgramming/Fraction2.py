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

a = Fraction() # a = 0
Fraction.set_val(a,3,7) # 重新設定為 3/7
print( a.mul(2) ) # 輸出 6/7

# 以下 b 名稱皆為第一次使用
Fraction.__init__(b,3,7) # 錯誤，b 未定義
Fraction(b,3,7) # 錯誤，b 未定義