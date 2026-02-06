# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

cno = "零一二三四五六七八九"

# 基礎類別：多邊形類別
class Polygon :

    def __init__( self , n ) : 
        self.npt = n

    def __str__( self ) :
        return cno[self.npt] + "邊形"

# 三角形繼承自多邊形
class Triangle(Polygon) :
    
    def __init__( self ) :
        Polygon.__init__(self,3) # 執行 Polygon 起始方法
    
    def __str__( self ) : 
        #return cno[self.npt] + "邊形"
        return "三角形"

# 四邊形繼承自多邊形
class Quadrangle(Polygon) :

    def __init__( self ) :
        Polygon.__init__(self,4) # 執行 Polygon 起始方法

class Rectangle(Quadrangle) :
    
    def __init__( self ) :
        Quadrangle.__init__(self) # 執行 Quadrangle 起始方法
    
    def __str__( self ) : 
        return "矩形"

if __name__ == "__main__" :
    # 四個不同圖形
    shapes = [ Polygon(5) , Triangle() , Quadrangle() , Rectangle() ]
    # 輸出：五邊形 三角形 四邊形 矩形 共四列
    for shape in shapes :
        print( shape ) # 等同 print( str(shape) )