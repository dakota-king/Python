# Object-Oriented Programming (OOP) - Classes and Objects
# OOP is a powerful way to organize and structure your code!

# =============================================================================
# 1. BASIC CLASS DEFINITION
# =============================================================================

class Dog:
    # Class attribute (shared by all instances)
    species = "Canis lupus"
    
    # Constructor method (__init__)
    def __init__(self, name, age, breed):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
        self.breed = breed
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    def get_info(self):
        return f"{self.name} is a {self.age}-year-old {self.breed}"

# Create objects (instances of the class)
dog1 = Dog("Buddy", 3, "Golden Retriever")
dog2 = Dog("Max", 5, "German Shepherd")

print(dog1.bark())
print(dog2.get_info())
print(f"Species: {Dog.species}")

# =============================================================================
# 2. MORE COMPLEX CLASS EXAMPLE
# =============================================================================

class BankAccount:
    def __init__(self, account_holder, initial_balance=0):
        self.account_holder = account_holder
        self.balance = initial_balance
        self.transaction_history = []
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            self.transaction_history.append(f"Deposited ${amount}")
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            self.transaction_history.append(f"Withdrew ${amount}")
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Invalid withdrawal amount or insufficient funds"
    
    def get_balance(self):
        return f"Current balance: ${self.balance}"
    
    def get_statement(self):
        print(f"\n--- Account Statement for {self.account_holder} ---")
        for transaction in self.transaction_history:
            print(f"  {transaction}")
        print(f"Current Balance: ${self.balance}")

# Using the BankAccount class
account = BankAccount("Alice Johnson", 1000)
print(account.deposit(500))
print(account.withdraw(200))
print(account.get_balance())
account.get_statement()

# =============================================================================
# 3. INHERITANCE - CREATING SUBCLASSES
# =============================================================================

# Parent class (Base class)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return f"{self.name} makes a sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

# Child class (Derived class)
class Cat(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Feline")  # Call parent constructor
        self.breed = breed
    
    # Override parent method
    def make_sound(self):
        return f"{self.name} says Meow!"
    
    # Add new method specific to Cat
    def purr(self):
        return f"{self.name} is purring contentedly"

class Bird(Animal):
    def __init__(self, name, can_fly=True):
        super().__init__(name, "Avian")
        self.can_fly = can_fly
    
    def make_sound(self):
        return f"{self.name} chirps melodiously"
    
    def fly(self):
        if self.can_fly:
            return f"{self.name} soars through the sky"
        return f"{self.name} cannot fly"

# Using inheritance
cat = Cat("Whiskers", "Persian")
bird = Bird("Robin")
penguin = Bird("Pingu", can_fly=False)

print(cat.info())
print(cat.make_sound())
print(cat.purr())

print(bird.make_sound())
print(bird.fly())

print(penguin.make_sound())
print(penguin.fly())

# =============================================================================
# 4. CLASS METHODS AND STATIC METHODS
# =============================================================================

class MathUtils:
    pi = 3.14159
    
    def __init__(self, name):
        self.name = name
    
    # Instance method (needs self)
    def greet(self):
        return f"Hello from {self.name}"
    
    # Class method (works with the class, not instance)
    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2
    
    # Static method (doesn't need class or instance)
    @staticmethod
    def add_numbers(a, b):
        return a + b

# Using different types of methods
math_obj = MathUtils("Calculator")
print(math_obj.greet())                    # Instance method

print(MathUtils.circle_area(5))            # Class method
print(MathUtils.add_numbers(10, 20))       # Static method

# =============================================================================
# 5. SPECIAL METHODS (MAGIC METHODS)
# =============================================================================

class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    # String representation
    def __str__(self):
        return f"'{self.title}' by {self.author}"
    
    # Detailed representation (for debugging)
    def __repr__(self):
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    # Length of the book
    def __len__(self):
        return self.pages
    
    # Comparison methods
    def __eq__(self, other):
        return self.pages == other.pages
    
    def __lt__(self, other):
        return self.pages < other.pages

# Using special methods
book1 = Book("Python Programming", "John Smith", 300)
book2 = Book("Web Development", "Jane Doe", 250)

print(book1)                    # Uses __str__
print(repr(book1))              # Uses __repr__
print(len(book1))               # Uses __len__
print(book1 == book2)           # Uses __eq__
print(book1 > book2)            # Uses __lt__ (reversed)

# =============================================================================
# 6. PRACTICAL EXAMPLE - STUDENT MANAGEMENT SYSTEM
# =============================================================================

class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {}
        self.courses = []
    
    def enroll_course(self, course):
        if course not in self.courses:
            self.courses.append(course)
            self.grades[course] = []
            return f"{self.name} enrolled in {course}"
        return f"{self.name} is already enrolled in {course}"
    
    def add_grade(self, course, grade):
        if course in self.courses:
            if 0 <= grade <= 100:
                self.grades[course].append(grade)
                return f"Grade {grade} added for {course}"
            return "Grade must be between 0 and 100"
        return f"Student not enrolled in {course}"
    
    def get_average(self, course):
        if course in self.grades and self.grades[course]:
            return sum(self.grades[course]) / len(self.grades[course])
        return 0
    
    def get_overall_average(self):
        all_grades = []
        for course_grades in self.grades.values():
            all_grades.extend(course_grades)
        
        if all_grades:
            return sum(all_grades) / len(all_grades)
        return 0
    
    def get_transcript(self):
        print(f"\n--- Transcript for {self.name} (ID: {self.student_id}) ---")
        for course in self.courses:
            avg = self.get_average(course)
            print(f"{course}: Average = {avg:.1f}")
        print(f"Overall Average: {self.get_overall_average():.1f}")

# Using the Student class
student = Student("Emma Wilson", "S12345")
print(student.enroll_course("Python Programming"))
print(student.enroll_course("Data Structures"))
print(student.enroll_course("Web Development"))

print(student.add_grade("Python Programming", 95))
print(student.add_grade("Python Programming", 88))
print(student.add_grade("Data Structures", 92))
print(student.add_grade("Web Development", 90))

student.get_transcript()

# =============================================================================
# 7. PRACTICE EXERCISES
# =============================================================================

print("\n" + "="*50)
print("PRACTICE TIME!")
print("="*50)

# Exercise 1: Create a simple Calculator class
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} × {b} = {result}")
        return result
    
    def get_history(self):
        return self.history

calc = Calculator()
print(f"5 + 3 = {calc.add(5, 3)}")
print(f"4 × 7 = {calc.multiply(4, 7)}")
print(f"History: {calc.get_history()}")

# Exercise 2: Create a simple Car class
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.mileage = 0
    
    def drive(self, miles):
        self.mileage += miles
        return f"Drove {miles} miles. Total mileage: {self.mileage}"
    
    def __str__(self):
        return f"{self.year} {self.make} {self.model}"

car = Car("Toyota", "Camry", 2020)
print(car)
print(car.drive(100))
print(car.drive(50))

print("\n🎉 Fantastic! You've learned Object-Oriented Programming!")
print("Next: Run this file and then move to 05_file_handling.py")
