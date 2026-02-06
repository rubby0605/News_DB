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

    # 雙底線開始的方法名稱，不能由物件直接取用
    def __inverse( self ) :
        return Fraction(self.den,self.num)
    
    # 雙底線開始的方法名稱可由類別其他方法使用
    def inv( self ) :
        return self.__inverse()

a = Fraction(2,3)
# 錯誤，無此方法
# print ( a.__inverse () )
# # 正確，原雙底線方法名稱前被加上 _Fraction
# print ( a._Fraction__inverse () )
# # 正確
print ( a.inv() ) # 輸出：3/2 