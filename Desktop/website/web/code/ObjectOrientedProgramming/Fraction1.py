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

a = Fraction() # a = 0 參考第 1 頁
a.set_val(2,5) # 重新設定為 2/5

print( a.get_num() ) # 印出 a 的分子 2
#a.set_val() # 錯誤，少了分子參數

b = Fraction()
b.num , b.den = 3, 4 # 直接設定 b 物件分子與分母
print( b ) # 印出：3/4