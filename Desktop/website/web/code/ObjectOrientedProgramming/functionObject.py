# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def my_func():
   print('My function was called')
   
def second_func():
   print('Second function was called')

def another_func(func):
   print('The name: ', end=' ')
   print(func.__name__)
   print('The class:', end=' ')
   print(func.__class__)
   print("Now I'll call the function passed in")
   func()

another_func(my_func)
another_func(second_func)


