# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

cno = "零一二三四五六七八九"

# 繼承順序：Polygon
class Polygon :
    def __init__( self , n ) : 
        self.npt = n
    def name( self ) : 
        return cno[self.npt] + "邊形"
    def total_angle( self ) : 
        return 180*(self.npt-2)
    def property( self ) :
        return "{}個邊，內角和 {} 度".format(cno[self.npt],self.total_angle())
    def __str__( self ) : 
        return self.name()

# 繼承順序：Triangle、Polygon
class Triangle(Polygon) :
    def __init__( self ) : 
        super().__init__(3)
    def name( self ) : 
        return "三角形"
    def property( self ) : 
        return ( super().property() + "，有內心、外心、垂心、重心、旁心" )
    def __str__( self ) : 
        return self.name()

# 繼承順序：Quadrangle、Polygon
class Quadrangle(Polygon) :
    def __init__( self ) : 
        Polygon.__init__(self,4)
    # 可省略
    def __str__( self ) : 
        return self.name()

# 繼承順序：Rectangle、Quadrangle、Polygon
class Rectangle(Quadrangle) :
    def __init__( self ) : 
        super().__init__()
    def name( self ) : 
        return "矩形"
    def property( self ) : 
        return ( Polygon.property(self) + " ，兩對邊等長，四個角皆為直角" )
    def __str__( self ) : 
        return self.name()

if __name__ == "__main__" :
    shapes = [ Polygon(5) , Triangle() , Quadrangle() , Rectangle() ]
    for shape in shapes :
        print(shape,'：',shape.property(),sep="")
    