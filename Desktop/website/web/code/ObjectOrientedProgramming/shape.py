# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

cno = "零一二三四五六七八九"

# 平面點類別
class Point :
    def __init__( self , x = 0 , y = 0 ) : 
        self.x , self.y = x , y
    
    def __str__( self ) : 
        return "({},{})".format(self.x,self.y)

# 多邊形類別
class Polygon :
    def __init__( self , pts ) : 
        self.pts = pts
    
    def __str__( self ) : 
        return " ".join( [ str(pt) for pt in self.pts ] )
    
    def name( self ) : 
        return cno[len(self.pts)] + "邊形"

# 三角形：使用多邊形的起始設定方法
class Triangle(Polygon) :
    def __init__( self , p1 , p2 , p3 ) : 
        Polygon.__init__(self,[p1,p2,p3])
    
    def name( self ) : 
        return "三角形"
    
if __name__ == "__main__" :
    # 定義四個點與兩個物件
    p1 , p2 , p3 , p4 = Point(0,0) , Point(1,0) , Point(0,2) , Point(-2,2)
    poly , tri = Polygon([p1,p2,p3,p4]) , Triangle(p1,p2,p3)
    for foo in [ poly , tri ] : print(foo.name(),str(foo))
