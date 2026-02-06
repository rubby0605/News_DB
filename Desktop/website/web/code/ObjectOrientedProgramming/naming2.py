# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

class MyClass():                
    def __init__(self, a="Hello"):
        self.__a=a
    
    @property               # 修飾器, 將方法 a() 修飾為物件屬性
    def a(self):             
        return self.__a     # 傳回私有屬性之 值

myobj=MyClass()

print(dir(myobj))           # 檢視物件內容    

myobj.a                     # 讀取私有屬性值

#myobj.a()                   # 被修飾器綁定為屬性後無法被呼叫 (not callable)
  
#del myobj.a                 # 無法刪除私有屬性
    
myobj.a="World"             # 無法設定私有屬性之值
