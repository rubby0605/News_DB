# -*- coding: utf-8 -*-
"""
Created on Mon May 31 08:29:47 2021

@author: Alvin
"""

class MyClass(object):
   def setAge(self, num):
      self.age = num

   def getAge(self):
      return self.age

## 範例化物件
zack = MyClass()
zack.setAge(45)
print(zack.getAge())
zack.setAge("Fourty Five")
print(zack.getAge())