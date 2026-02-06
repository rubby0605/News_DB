# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class Employee:
   'Common base class for all employees'
   empCount = 0

   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1

   def displayCount(self):
     print("Total Employee %d" % Employee.empCount)

   def displayEmployee(self):
      print("Name : ", self.name,  ", Salary: ", self.salary)
      
## This would create first object of Employee class
emp1 = Employee("Maxsu", 2000)
## This would create second object of Employee class
emp2 = Employee("Kobe", 5000)

emp1.displayEmployee()
emp2.displayEmployee()
print("Total Employee %d" % Employee.empCount)

emp1.salary = 7000  # Add an 'salary' attribute.
emp1.name = 'xyz'  # Modify 'age' attribute.
#del emp1.salary  # Delete 'age' attribute.

'''
hasattr(emp1, 'salary')         # Returns true if 'salary' attribute exists
getattr(emp1, 'salary')         # Returns value of 'salary' attribute
setattr(emp1, 'salary', 7000)   # Set attribute 'salary' at 7000
delattr(emp1, 'salary')         # Delete attribute 'salary'
'''


