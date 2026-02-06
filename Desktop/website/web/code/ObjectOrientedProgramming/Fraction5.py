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

    # 由字串轉換來的分數
    @classmethod
    def fromstr( cls , fstr ) :
        if fstr.isdigit() :
            num , den = int(fstr) , 1
        else :
            num , den = map( int , fstr.split('/') )
        return cls(num,den)

    # 帶分數型式
    @classmethod
    def mixed_fraction( cls , a = 0 , n = 0 , d = 1 ) :
        num , den = a * d + n , d
        return cls(num,den)

    # 分數資料說明
    @classmethod
    def data_doc( cls ) :
        return "num:分子 , den:分母"

# 以下三個 Fraction 被自動設為類別方法的第一個參數
a = Fraction.fromstr("5")
b = Fraction.fromstr("4/7")
c = Fraction.mixed_fraction(2,3,4)
# 印出：5 4/7 11/4
print( a , b , c )
# 印出：num:分子 , den:分母
print( Fraction.data_doc() )