# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 06:54:07 2021

@author: Alvin
"""

class GetSet(object):

   instance_count = 0 # public

   __mangled_name = 'no privacy!' # special variable

   def __init__(self, value):
      self._attrval = value # _attrval is for internal use only
      GetSet.instance_count += 1

   @property
   def var(self):
      print('Getting the "var" attribute')
      return self._attrval

   @var.setter
   def var(self, value):
      print('setting the "var" attribute')
      self._attrval = value

   @var.deleter
   def var(self):
      print('deleting the "var" attribute')
      self._attrval = None

cc = GetSet(5)
cc.var = 10 # public name
print(cc.var)
print(cc._attrval)
print(cc._GetSet__mangled_name)
del cc.var


