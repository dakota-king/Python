# 🐍 ABSOLUTE BEGINNER'S PYTHON GUIDE
# Start here if you've never programmed before!
# This file explains EVERYTHING step by step

print("🎉 Welcome to Python Programming!")
print("Let's start from the very beginning...")

# =============================================================================
# WHAT IS PROGRAMMING?
# =============================================================================

# Programming is like giving instructions to a computer
# Just like you might give directions to a friend:
# "Go straight, turn left, stop at the red house"
# 
# In programming, we give the computer step-by-step instructions
# The computer follows these instructions exactly as written

# =============================================================================
# 1. YOUR FIRST PYTHON COMMAND - PRINT
# =============================================================================

# The print() function displays text on your screen
# Think of it like the computer "saying" something out loud

print("Hello, World!")  # This makes the computer display: Hello, World!

# Let's try more examples:
print("My name is Python")
print("I love learning!")
print("This is fun!")

# You can print numbers too:
print(42)
print(3.14)

# EXERCISE: Try changing the text in the quotes above and run this file again!

# =============================================================================
# 2. COMMENTS - NOTES FOR HUMANS
# =============================================================================

# Anything that starts with # is a "comment"
# Comments are notes for humans - the computer ignores them
# Use comments to explain what your code does

print("This line runs")  # This is a comment - it doesn't run
# print("This line won't run because it starts with #")

# Good comments explain WHY you're doing something:
print("Welcome!")  # Greeting the user when program starts

# =============================================================================
# 3. VARIABLES - STORING INFORMATION
# =============================================================================

# Variables are like boxes that store information
# You give the box a name, and put something inside it

# Creating variables (putting information in boxes):
my_name = "Alice"           # This box is called "my_name" and contains "Alice"
my_age = 25                 # This box is called "my_age" and contains 25
my_height = 5.6             # This box is called "my_height" and contains 5.6
is_student = True           # This box is called "is_student" and contains True

# Now we can use these variables:
print("My name is:")
print(my_name)              # This will show: Alice

print("My age is:")
print(my_age)               # This will show: 25

# We can use variables in sentences:
print("Hello, my name is", my_name)
print("I am", my_age, "years old")

# EXERCISE: Create variables for your own name and age, then print them

# =============================================================================
# 4. DIFFERENT TYPES OF DATA
# =============================================================================

# Python can store different types of information:

# TEXT (called "strings" - think of a string of letters)
favorite_color = "blue"
favorite_food = "pizza"
city = "New York"

# NUMBERS (two types)
whole_number = 42           # Called "integer" or "int"
decimal_number = 3.14       # Called "float" (floating point)

# TRUE/FALSE (called "boolean" or "bool")
likes_pizza = True          # True or False (notice the capital letters!)
is_raining = False

# Let's see what type each variable is:
print("Type of favorite_color:", type(favorite_color))
print("Type of whole_number:", type(whole_number))
print("Type of likes_pizza:", type(likes_pizza))

# =============================================================================
# 5. WORKING WITH TEXT (STRINGS)
# =============================================================================

# Strings are text surrounded by quotes
first_name = "John"
last_name = "Smith"

# You can combine strings with + (called "concatenation")
full_name = first_name + " " + last_name
print("Full name:", full_name)

# Better way to combine text and variables (called "f-strings"):
greeting = f"Hello, {first_name}! Nice to meet you."
print(greeting)

age = 30
message = f"My name is {first_name} and I am {age} years old."
print(message)

# Useful string operations:
text = "  Python Programming  "
print("Original text:", text)
print("Without spaces:", text.strip())        # Removes spaces from ends
print("Lowercase:", text.lower())              # Makes everything lowercase
print("Uppercase:", text.upper())              # Makes everything uppercase
print("Length:", len(text))                    # Counts characters

# EXERCISE: Create variables for your first and last name, 
# then use an f-string to introduce yourself

# =============================================================================
# 6. WORKING WITH NUMBERS
# =============================================================================

# Basic math operations:
a = 10
b = 3

print("Addition:", a + b)          # 10 + 3 = 13
print("Subtraction:", a - b)       # 10 - 3 = 7
print("Multiplication:", a * b)    # 10 * 3 = 30
print("Division:", a / b)          # 10 / 3 = 3.333...
print("Power:", a ** b)            # 10 to the power of 3 = 1000

# You can store the results:
sum_result = a + b
print("The sum is:", sum_result)

# You can do math with variables:
price = 19.99
tax_rate = 0.08
tax = price * tax_rate
total = price + tax
print(f"Price: ${price}")
print(f"Tax: ${tax:.2f}")          # .2f means show 2 decimal places
print(f"Total: ${total:.2f}")

# EXERCISE: Calculate the area of a rectangle with width=5 and height=8

# =============================================================================
# 7. LISTS - STORING MULTIPLE ITEMS
# =============================================================================

# Lists are like boxes that can hold multiple items
# Think of a shopping list or a to-do list

# Creating a list (items go in square brackets, separated by commas):
fruits = ["apple", "banana", "orange", "grape"]
numbers = [1, 2, 3, 4, 5]
mixed_list = ["hello", 42, True, 3.14]  # Lists can hold different types!

print("Fruits list:", fruits)
print("Numbers list:", numbers)

# Getting items from a list (counting starts at 0!):
print("First fruit:", fruits[0])       # apple (position 0)
print("Second fruit:", fruits[1])      # banana (position 1)
print("Last fruit:", fruits[-1])       # grape (negative numbers count from end)

# Adding items to a list:
fruits.append("strawberry")            # Adds to the end
print("After adding strawberry:", fruits)

# Finding out how many items are in a list:
print("Number of fruits:", len(fruits))

# EXERCISE: Create a list of your 3 favorite movies and print the first one

# =============================================================================
# 8. DICTIONARIES - STORING RELATED INFORMATION
# =============================================================================

# Dictionaries are like address books - they store pairs of information
# Each piece of information has a "key" (name) and a "value" (the information)

# Creating a dictionary (uses curly brackets):
person = {
    "name": "Alice",
    "age": 30,
    "city": "Boston",
    "job": "Teacher"
}

print("Person dictionary:", person)

# Getting information from a dictionary:
print("Name:", person["name"])         # Gets the value for "name"
print("Age:", person["age"])           # Gets the value for "age"

# Adding new information:
person["email"] = "alice@email.com"
print("After adding email:", person)

# Another example - a book:
book = {
    "title": "Harry Potter",
    "author": "J.K. Rowling", 
    "pages": 309,
    "genre": "Fantasy"
}

print(f"The book '{book['title']}' by {book['author']} has {book['pages']} pages.")

# EXERCISE: Create a dictionary about yourself with name, age, and favorite color

# =============================================================================
# 9. GETTING INPUT FROM THE USER
# =============================================================================

# The input() function lets you ask the user to type something
# Uncomment the lines below to try them (remove the # at the beginning):

# user_name = input("What's your name? ")
# print(f"Nice to meet you, {user_name}!")

# user_age = input("How old are you? ")
# print(f"You are {user_age} years old!")

# Note: input() always gives you text, even if the user types a number
# To convert to a number, use int() or float():

# age_text = input("Enter your age: ")
# age_number = int(age_text)  # Converts text to whole number
# print(f"Next year you'll be {age_number + 1}")

print("(Input examples are commented out so the program runs automatically)")

# =============================================================================
# 10. PRACTICE EXERCISES WITH STEP-BY-STEP SOLUTIONS
# =============================================================================

print("\n" + "="*60)
print("🏋️ PRACTICE TIME! Let's solve some problems together")
print("="*60)

# EXERCISE 1: Create a simple profile
print("\n📝 Exercise 1: Creating a Profile")
print("-" * 30)

# Step 1: Create variables
student_name = "Emma"
student_age = 20
student_major = "Computer Science"
student_gpa = 3.8

# Step 2: Display the profile
print("STUDENT PROFILE:")
print(f"Name: {student_name}")
print(f"Age: {student_age}")
print(f"Major: {student_major}")
print(f"GPA: {student_gpa}")

# EXERCISE 2: Calculate a tip
print("\n💰 Exercise 2: Tip Calculator")
print("-" * 30)

# Step 1: Set up the bill
bill_amount = 45.50
tip_percentage = 18  # 18% tip

# Step 2: Calculate tip and total
tip_amount = bill_amount * (tip_percentage / 100)
total_amount = bill_amount + tip_amount

# Step 3: Display results
print(f"Bill amount: ${bill_amount}")
print(f"Tip ({tip_percentage}%): ${tip_amount:.2f}")
print(f"Total: ${total_amount:.2f}")

# EXERCISE 3: Working with a shopping list
print("\n🛒 Exercise 3: Shopping List Manager")
print("-" * 30)

# Step 1: Create a shopping list
shopping_list = ["milk", "bread", "eggs", "apples"]

# Step 2: Display the list
print("Original shopping list:")
for i, item in enumerate(shopping_list):
    print(f"{i + 1}. {item}")  # enumerate gives us position numbers

# Step 3: Add an item
shopping_list.append("cheese")
print("\nAfter adding cheese:")
for i, item in enumerate(shopping_list):
    print(f"{i + 1}. {item}")

# Step 4: Show list statistics
print(f"\nTotal items: {len(shopping_list)}")
print(f"First item: {shopping_list[0]}")
print(f"Last item: {shopping_list[-1]}")

# EXERCISE 4: Create a simple gradebook
print("\n📚 Exercise 4: Simple Gradebook")
print("-" * 30)

# Step 1: Create a student record
student = {
    "name": "Alex",
    "grades": [85, 92, 78, 96, 88],
    "subject": "Math"
}

# Step 2: Calculate average grade
total_points = sum(student["grades"])  # sum() adds all numbers in a list
num_grades = len(student["grades"])
average = total_points / num_grades

# Step 3: Display results
print(f"Student: {student['name']}")
print(f"Subject: {student['subject']}")
print(f"Grades: {student['grades']}")
print(f"Average: {average:.1f}")

# Determine letter grade
if average >= 90:
    letter_grade = "A"
elif average >= 80:
    letter_grade = "B"
elif average >= 70:
    letter_grade = "C"
elif average >= 60:
    letter_grade = "D"
else:
    letter_grade = "F"

print(f"Letter Grade: {letter_grade}")

# =============================================================================
# 11. WHAT YOU'VE LEARNED
# =============================================================================

print("\n" + "="*60)
print("🎉 CONGRATULATIONS! You've learned the basics!")
print("="*60)

skills_learned = [
    "✅ How to use print() to display information",
    "✅ How to write comments to explain your code",
    "✅ How to create variables to store information",
    "✅ The different types of data (text, numbers, True/False)",
    "✅ How to work with text (strings)",
    "✅ How to do basic math operations",
    "✅ How to create lists to store multiple items",
    "✅ How to create dictionaries to store related information",
    "✅ How to get input from users",
    "✅ How to solve simple programming problems"
]

print("\n🏆 SKILLS YOU NOW HAVE:")
for skill in skills_learned:
    print(f"  {skill}")

print("\n🚀 WHAT'S NEXT?")
next_steps = [
    "1. Practice these concepts by modifying the examples above",
    "2. Try the exercises at the end of each section",
    "3. Move on to 01_python_basics.py for more advanced topics",
    "4. Don't worry if something doesn't make sense - keep practicing!"
]

for step in next_steps:
    print(f"  {step}")

print("\n💡 REMEMBER:")
print("  • Programming is like learning a new language - it takes practice!")
print("  • Don't be afraid to experiment and make mistakes")
print("  • Every programmer started exactly where you are now")
print("  • The more you practice, the easier it becomes")

print("\n🐍 You're doing great! Keep going!")
print("Next file: 01_python_basics.py")
