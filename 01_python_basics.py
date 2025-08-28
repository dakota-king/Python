# Python Basics - Learn Fast! (Detailed Explanations)
# This file covers essential Python concepts with step-by-step explanations
# 🔴 If you're completely new to programming, start with 00_absolute_beginner_start.py first!

# =============================================================================
# 1. VARIABLES AND DATA TYPES
# =============================================================================

# Variables (no need to declare type)
name = "Python Learner"
age = 25
height = 5.9
is_learning = True

print(f"Hello {name}! You are {age} years old.")

# Basic Data Types
text = "Hello World"           # str (string)
number = 42                    # int (integer)
decimal = 3.14                 # float
boolean = True                 # bool (True/False)
nothing = None                 # NoneType

# Check type of variable
print(type(name))              # <class 'str'>
print(type(age))               # <class 'int'>

# =============================================================================
# 2. STRINGS - SUPER IMPORTANT!
# =============================================================================

# String operations
first_name = "John"
last_name = "Doe"

# String concatenation
full_name = first_name + " " + last_name
print(full_name)

# F-strings (modern and preferred way)
greeting = f"Hello, {first_name}! Welcome to Python."
print(greeting)

# String methods (very useful!)
text = "  Python Programming  "
print(text.strip())            # Remove whitespace
print(text.lower())            # Lowercase
print(text.upper())            # Uppercase
print(text.replace("Python", "Java"))  # Replace text
print(len(text))               # Length of string

# =============================================================================
# 3. LISTS - LIKE ARRAYS BUT BETTER
# =============================================================================

# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, True, 3.14]  # Can mix types!

# Accessing elements (0-based indexing)
print(fruits[0])               # First element: "apple"
print(fruits[-1])              # Last element: "orange"

# List methods
fruits.append("grape")         # Add to end
fruits.insert(1, "mango")      # Insert at position
fruits.remove("banana")        # Remove specific item
print(len(fruits))             # Length of list

# List slicing (VERY POWERFUL!)
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[2:5])            # Elements 2-4: [2, 3, 4]
print(numbers[:3])             # First 3: [0, 1, 2]
print(numbers[7:])             # From 7 to end: [7, 8, 9]
print(numbers[::2])            # Every 2nd: [0, 2, 4, 6, 8]

# =============================================================================
# 4. DICTIONARIES - KEY-VALUE PAIRS
# =============================================================================

# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "JavaScript", "SQL"]
}

# Accessing values
print(person["name"])          # "Alice"
print(person.get("age"))       # 30 (safer method)

# Adding/updating
person["email"] = "alice@email.com"
person["age"] = 31

# Dictionary methods
print(person.keys())           # All keys
print(person.values())         # All values
print(person.items())          # Key-value pairs

# =============================================================================
# 5. BASIC INPUT/OUTPUT
# =============================================================================

# Getting user input
# user_name = input("What's your name? ")
# user_age = int(input("What's your age? "))  # Convert to integer
# print(f"Nice to meet you, {user_name}! You are {user_age} years old.")

# =============================================================================
# 6. QUICK PRACTICE EXERCISES
# =============================================================================

print("\n" + "="*50)
print("PRACTICE TIME!")
print("="*50)

# Exercise 1: Create a list of your favorite movies
movies = ["The Matrix", "Inception", "Interstellar"]
print(f"My favorite movies: {movies}")

# Exercise 2: Create a dictionary about yourself
me = {
    "name": "Python Student",
    "learning": "Python",
    "goal": "Become a developer",
    "progress": "Just started!"
}
print(f"About me: {me}")

# Exercise 3: String manipulation
message = "python is awesome"
print(f"Original: {message}")
print(f"Capitalized: {message.title()}")
print(f"Length: {len(message)}")
print(f"Contains 'python': {'python' in message}")

print("\n🎉 Great job! You've learned Python basics!")
print("Next: Run this file and then move to 02_control_structures.py")
