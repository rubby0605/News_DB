# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:47:51 2021

@author: teacher
"""

class Animal(object):
    def __init__(self, name):
        self.name = name

    def eat(self, food):
        print('%s is eating %s , '%(self.name, food))

class Dog(Animal):

    def fetch(self, thing):
        print("{0} wags {1}".format(self.name, thing))

    def show_affection(self):
        print("{0} wags tail ".format(self.name))

class Cat(Animal):

    def swatstring(self):
        print('%s shreds the string! ' % (self.name))

    def show_affection(self):
        print("{0} purrs ".format(self.name))

d = Dog('Ranger')
c = Cat("Meow")

d.show_affection()
c.show_affection()