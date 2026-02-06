# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
class Human:
   def sayHello(self, name = None):
      if name is not None:
         print('Hello ' + name)
      else:
         print('Hello ')

#Create Instance
obj = Human()

#Call the method, else part will be executed
obj.sayHello()

#Call the method with a parameter, if part will be executed
obj.sayHello('Rahul')


