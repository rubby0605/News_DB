# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

class MyClass():
    def __init__(self, a="hello", b="world"):     # 設定參數之預設值
        self._a=a              # 被保護的屬性
        self.__b=b             # 私有的屬性
    def showInfo(self):
        print("a =", self._a, "b =", self.__b)    # 透過 self 物件存取其屬性
        
myobj=MyClass()
print(myobj._a)     # 外部程式碼仍可存取被保護的屬性

myobj._a=123        # 被保護的屬性可以被更改
print(myobj._a)
myobj.showInfo()    # 屬性 _a 真的被改變了

print(myobj.__b)    # 外部程式碼無法存取私有的屬性
myobj.showInfo()    # 私有屬性只能透過公開方法取得  

#如果直接改變私有屬性之值, 雖然不會出現錯誤, 但其實並沒有真的被改變 :
myobj.__b=456       # 企圖更改私有屬性之值
print(myobj.__b)    # 檢視似乎值被改變了

myobj.showInfo()     # 呼叫公開方法 showInfo() 顯示並沒有被改變

