#!/usr/bin/env python << -- Mac users may need to chg to "python3"
import os, sys
'''
In Python, the fundamental difference between numbers and characters lies in their data types and how they are handled:
===================
1. Data Types:
===================
Numbers:
-------------------
Python has distinct numeric data types to represent different kinds of numbers:
Integers (int): Whole numbers (e.g., 5, -10, 0).
Floating-point numbers (float): Numbers with decimal points (e.g., 3.14, -0.5, 2.0).
Complex numbers (complex): Numbers with a real and an imaginary part (e.g., 2 + 3j).
------------------
Characters:
------------------
Python does not have a separate "character" data type. Instead, individual characters are treated as strings of length one (str). For example, 'a', '5', '!' are all considered strings.

===================
2. Operations and Behavior:
===================
Numbers:
-------------------
Numeric data types support mathematical operations like addition, subtraction, multiplication, division, exponentiation, etc.
Python
'''
num1 = 10
num2 = 3
result = num1 / num2  # result will be 3.333... (float)
'''
-------------------
Characters (Strings):
-------------------
Strings, even single-character ones, primarily support string-specific operations like concatenation, slicing, case conversion, and searching.
Python
'''
char1 = 'A'
char2 = 'B'
combined = char1 + char2  # combined will be 'AB'
'''
While a string might contain a numeric character (e.g., '5'), it cannot be directly used in mathematical operations without type conversion.
===================
3. Type Conversion:
===================
Python provides built-in functions to convert between numeric types and strings:
int(): Converts a string or float to an integer.
float(): Converts a string or integer to a float.
str(): Converts a number to its string representation.
Python
'''
string_num = '123'
integer_num = int(string_num)  # integer_num is 123
print(f'{integer_num=} is datatype: {type(integer_num)=}')
'''
In summary, numbers in Python are distinct data types designed for mathematical computations, while characters are handled as strings, even when they represent single symbols, and require explicit type conversion for numeric operations.
'''
