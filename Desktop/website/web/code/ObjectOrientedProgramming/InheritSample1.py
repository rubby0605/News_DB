# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
class Date(object):
    def get_date(self):
        return "2018-06-30"

class Time(Date):
    def get_time(self):
        return "09:09:09"

dt = Date()
print("Get date from Date class: ", dt.get_date())

tm = Time()
print("Get time from Time class: ", tm.get_time())
print("Get date from class by inheriting or calling Date class method: ", tm.get_date())
