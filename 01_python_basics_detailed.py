# Python Basics - Detailed Step-by-Step Guide
# This version explains everything in much more detail for beginners

# =============================================================================
# WHY THIS FILE EXISTS
# =============================================================================

# This file takes the concepts from 00_absolute_beginner_start.py and goes deeper
# Each concept is explained with:
# 1. What it is and why it's useful
# 2. How to use it with examples
# 3. Common mistakes to avoid
# 4. Practice exercises

print("🐍 Python Basics - Detailed Version")
print("=" * 50)

# =============================================================================
# 1. VARIABLES - DEEPER UNDERSTANDING
# =============================================================================

print("\n📦 SECTION 1: VARIABLES (Storage Boxes for Information)")
print("-" * 50)

# WHAT ARE VARIABLES?
# Variables are like labeled boxes where you store information
# The label is the variable name, the contents is the value

# CREATING VARIABLES (Assignment)
name = "Sarah"              # Text goes in quotes
age = 28                    # Numbers don't need quotes
height = 5.7                # Decimal numbers are called "floats"
is_employed = True          # True/False values are called "booleans"

print("Variable examples:")
print(f"name = '{name}' (this is text/string)")
print(f"age = {age} (this is a whole number/integer)")
print(f"height = {height} (this is a decimal/float)")
print(f"is_employed = {is_employed} (this is True or False/boolean)")

# VARIABLE NAMING RULES
# ✅ Good variable names:
first_name = "John"         # Use underscores for multiple words
user_age = 25              # Be descriptive
total_cost = 49.99         # Tell us what the variable contains

# ❌ Bad variable names (but these still work):
x = "John"                 # Not descriptive
a = 25                     # What does 'a' mean?
n = "Sarah"                # Too short, unclear

print(f"\nGood naming example: first_name = '{first_name}'")
print("This tells us exactly what information is stored!")

# VARIABLES CAN CHANGE (That's why they're called "variable"!)
score = 0                  # Start with 0
print(f"Initial score: {score}")

score = 10                 # Change to 10
print(f"After first level: {score}")

score = score + 5          # Add 5 to current score
print(f"After bonus: {score}")

score += 3                 # Shortcut: same as score = score + 3
print(f"After another bonus: {score}")

# =============================================================================
# 2. DATA TYPES - WHAT KIND OF INFORMATION CAN WE STORE?
# =============================================================================

print("\n🏷️ SECTION 2: DATA TYPES (Different Kinds of Information)")
print("-" * 50)

# Python has several built-in data types. Here are the most important ones:

# 1. STRINGS (TEXT)
# Strings are sequences of characters (letters, numbers, symbols)
# They MUST be surrounded by quotes

student_name = "Alice Johnson"      # Double quotes work
teacher_name = 'Bob Smith'          # Single quotes work too
school_motto = "Learn, Grow, Succeed!"

print("String examples:")
print(f"Student: {student_name}")
print(f"Teacher: {teacher_name}")
print(f"Motto: {school_motto}")

# You can include quotes inside strings:
quote = "She said, 'Hello there!'"  # Single quotes inside double quotes
another_quote = 'He replied, "Nice to meet you!"'  # Double quotes inside single quotes
print(f"Quote example: {quote}")

# 2. INTEGERS (WHOLE NUMBERS)
# Integers are whole numbers (no decimal point)
students_in_class = 25
year = 2024
temperature = -5               # Can be negative

print(f"\nInteger examples:")
print(f"Students in class: {students_in_class}")
print(f"Year: {year}")
print(f"Temperature: {temperature}°F")

# 3. FLOATS (DECIMAL NUMBERS)
# Floats are numbers with decimal points
price = 19.99
pi = 3.14159
percentage = 85.5

print(f"\nFloat examples:")
print(f"Price: ${price}")
print(f"Pi: {pi}")
print(f"Percentage: {percentage}%")

# 4. BOOLEANS (TRUE/FALSE)
# Booleans can only be True or False (notice the capital letters!)
is_sunny = True
is_raining = False
has_homework = True

print(f"\nBoolean examples:")
print(f"Is it sunny? {is_sunny}")
print(f"Is it raining? {is_raining}")
print(f"Do I have homework? {has_homework}")

# CHECKING DATA TYPES
# Use type() to see what type of data you have
print(f"\nChecking data types:")
print(f"type(student_name) = {type(student_name)}")
print(f"type(students_in_class) = {type(students_in_class)}")
print(f"type(price) = {type(price)}")
print(f"type(is_sunny) = {type(is_sunny)}")

# =============================================================================
# 3. STRINGS - WORKING WITH TEXT (DETAILED)
# =============================================================================

print("\n📝 SECTION 3: STRINGS (Working with Text)")
print("-" * 50)

# CREATING STRINGS
first_name = "Emma"
last_name = "Wilson"
middle_initial = "J"

# STRING CONCATENATION (Joining strings together)
# Method 1: Using + operator
full_name_method1 = first_name + " " + middle_initial + ". " + last_name
print(f"Method 1 (+): {full_name_method1}")

# Method 2: Using f-strings (RECOMMENDED - easier to read!)
full_name_method2 = f"{first_name} {middle_initial}. {last_name}"
print(f"Method 2 (f-string): {full_name_method2}")

# Method 3: Using .format() (older way, but still used)
full_name_method3 = "{} {}. {}".format(first_name, middle_initial, last_name)
print(f"Method 3 (.format): {full_name_method3}")

# STRING METHODS (Built-in functions that work on strings)
message = "  Hello, World! Welcome to Python!  "
print(f"\nOriginal message: '{message}'")

# Common string methods:
print(f"Length: {len(message)} characters")
print(f"Uppercase: '{message.upper()}'")
print(f"Lowercase: '{message.lower()}'")
print(f"Title Case: '{message.title()}'")
print(f"Strip whitespace: '{message.strip()}'")
print(f"Replace 'World' with 'Python': '{message.replace('World', 'Python')}'")

# STRING SLICING (Getting parts of a string)
word = "Programming"
print(f"\nString slicing with '{word}':")
print(f"First character: '{word[0]}'")      # P (position 0)
print(f"Last character: '{word[-1]}'")      # g (position -1)
print(f"First 4 characters: '{word[0:4]}'") # Prog (positions 0-3)
print(f"Last 4 characters: '{word[-4:]}'")  # ming (last 4)
print(f"Every 2nd character: '{word[::2]}'") # Pormig (every 2nd)

# CHECKING STRING CONTENTS
email = "user@example.com"
print(f"\nChecking string contents for '{email}':")
print(f"Contains '@': {'@' in email}")
print(f"Starts with 'user': {email.startswith('user')}")
print(f"Ends with '.com': {email.endswith('.com')}")
print(f"Is all lowercase: {email.islower()}")

# =============================================================================
# 4. NUMBERS AND MATH OPERATIONS (DETAILED)
# =============================================================================

print("\n🔢 SECTION 4: NUMBERS AND MATH")
print("-" * 50)

# BASIC ARITHMETIC OPERATORS
a = 15
b = 4

print(f"Working with a = {a} and b = {b}:")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")              # Always gives a float
print(f"Floor Division: {a} // {b} = {a // b}")      # Gives whole number part
print(f"Remainder (Modulo): {a} % {b} = {a % b}")    # Gives remainder
print(f"Exponentiation: {a} ** {b} = {a ** b}")      # a to the power of b

# PRACTICAL MATH EXAMPLES
print(f"\nPractical examples:")

# Calculate tip
bill = 45.50
tip_rate = 0.18
tip = bill * tip_rate
total = bill + tip
print(f"Bill: ${bill}")
print(f"Tip (18%): ${tip:.2f}")  # .2f means 2 decimal places
print(f"Total: ${total:.2f}")

# Convert temperature
celsius = 25
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")

# Calculate compound interest
principal = 1000
rate = 0.05  # 5%
time = 3     # years
amount = principal * (1 + rate) ** time
print(f"${principal} at {rate*100}% for {time} years = ${amount:.2f}")

# WORKING WITH DIFFERENT NUMBER TYPES
integer_num = 10
float_num = 10.5

print(f"\nMixing integers and floats:")
print(f"Integer + Float: {integer_num} + {float_num} = {integer_num + float_num}")
print(f"Result type: {type(integer_num + float_num)}")  # Result is always float

# CONVERTING BETWEEN NUMBER TYPES
text_number = "42"
print(f"\nConverting types:")
print(f"Text: '{text_number}' (type: {type(text_number)})")
print(f"To integer: {int(text_number)} (type: {type(int(text_number))})")
print(f"To float: {float(text_number)} (type: {type(float(text_number))})")

# =============================================================================
# 5. LISTS - STORING MULTIPLE VALUES (DETAILED)
# =============================================================================

print("\n📋 SECTION 5: LISTS (Storing Multiple Items)")
print("-" * 50)

# CREATING LISTS
# Lists are ordered collections of items, surrounded by square brackets []

# Empty list
empty_list = []
print(f"Empty list: {empty_list}")

# List with items
fruits = ["apple", "banana", "orange", "grape"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, True, 3.14]  # Lists can hold different types!

print(f"Fruits: {fruits}")
print(f"Numbers: {numbers}")
print(f"Mixed: {mixed}")

# ACCESSING LIST ITEMS (Indexing)
# Lists use zero-based indexing (counting starts at 0)
print(f"\nAccessing items in {fruits}:")
print(f"Index 0 (first): {fruits[0]}")
print(f"Index 1 (second): {fruits[1]}")
print(f"Index -1 (last): {fruits[-1]}")
print(f"Index -2 (second to last): {fruits[-2]}")

# LIST SLICING (Getting multiple items)
print(f"\nSlicing {numbers}:")
print(f"First 3 items: {numbers[0:3]}")    # Items 0, 1, 2
print(f"Items 2-4: {numbers[2:5]}")        # Items 2, 3, 4
print(f"Last 2 items: {numbers[-2:]}")     # Last 2 items
print(f"Every 2nd item: {numbers[::2]}")   # Every 2nd item

# MODIFYING LISTS
shopping_list = ["milk", "bread", "eggs"]
print(f"\nOriginal shopping list: {shopping_list}")

# Adding items
shopping_list.append("cheese")              # Add to end
print(f"After append: {shopping_list}")

shopping_list.insert(1, "butter")           # Insert at position 1
print(f"After insert: {shopping_list}")

# Removing items
shopping_list.remove("bread")               # Remove specific item
print(f"After remove: {shopping_list}")

last_item = shopping_list.pop()             # Remove and return last item
print(f"Removed '{last_item}', list now: {shopping_list}")

# Changing items
shopping_list[0] = "almond milk"            # Change first item
print(f"After change: {shopping_list}")

# LIST METHODS AND PROPERTIES
grades = [85, 92, 78, 96, 88, 85]
print(f"\nWorking with grades: {grades}")
print(f"Length: {len(grades)} grades")
print(f"Highest grade: {max(grades)}")
print(f"Lowest grade: {min(grades)}")
print(f"Sum of grades: {sum(grades)}")
print(f"Average grade: {sum(grades) / len(grades):.1f}")
print(f"Count of 85s: {grades.count(85)}")

# Sorting
grades_copy = grades.copy()                 # Make a copy
grades_copy.sort()                          # Sort in place
print(f"Sorted grades: {grades_copy}")
print(f"Original grades: {grades}")         # Original unchanged

# =============================================================================
# 6. DICTIONARIES - STORING KEY-VALUE PAIRS (DETAILED)
# =============================================================================

print("\n📖 SECTION 6: DICTIONARIES (Key-Value Storage)")
print("-" * 50)

# WHAT ARE DICTIONARIES?
# Dictionaries store information in key-value pairs
# Like a real dictionary: word (key) -> definition (value)
# Or like a contact book: name (key) -> phone number (value)

# CREATING DICTIONARIES
# Empty dictionary
empty_dict = {}
print(f"Empty dictionary: {empty_dict}")

# Dictionary with data
student = {
    "name": "Alex Johnson",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.7,
    "is_enrolled": True
}

print(f"Student dictionary: {student}")

# ACCESSING DICTIONARY VALUES
print(f"\nAccessing values:")
print(f"Name: {student['name']}")
print(f"Age: {student['age']}")
print(f"GPA: {student['gpa']}")

# Safe way to access (won't crash if key doesn't exist)
print(f"Email: {student.get('email', 'Not provided')}")  # Default value

# MODIFYING DICTIONARIES
print(f"\nModifying dictionary:")
student["email"] = "alex@university.edu"   # Add new key-value pair
student["age"] = 21                        # Change existing value
print(f"After modifications: {student}")

# DICTIONARY METHODS
print(f"\nDictionary methods:")
print(f"Keys: {list(student.keys())}")        # All keys
print(f"Values: {list(student.values())}")    # All values
print(f"Items: {list(student.items())}")      # All key-value pairs

# REAL-WORLD EXAMPLE: Inventory system
inventory = {
    "apples": 50,
    "bananas": 30,
    "oranges": 25,
    "grapes": 40
}

print(f"\nStore inventory: {inventory}")

# Sell some apples
inventory["apples"] -= 10  # Same as inventory["apples"] = inventory["apples"] - 10
print(f"After selling 10 apples: {inventory}")

# Add new fruit
inventory["strawberries"] = 20
print(f"After adding strawberries: {inventory}")

# Check if item exists
if "mangoes" in inventory:
    print(f"We have {inventory['mangoes']} mangoes")
else:
    print("We don't have mangoes in stock")

# =============================================================================
# 7. INPUT AND OUTPUT (DETAILED)
# =============================================================================

print("\n💬 SECTION 7: INPUT AND OUTPUT")
print("-" * 50)

# OUTPUT WITH PRINT()
# We've been using print() throughout this file
# Here are more advanced ways to use it:

name = "Python"
version = 3.12
year = 2024

# Different ways to print:
print("Simple text")
print("Name:", name)                        # Multiple items
print(f"Welcome to {name} {version}!")      # f-string (recommended)
print("Year: " + str(year))                 # Concatenation (convert number to string)

# Print formatting
price = 123.456
print(f"Price: ${price:.2f}")               # 2 decimal places
print(f"Price: ${price:10.2f}")             # 10 characters wide, 2 decimal places
print(f"Price: ${price:<10.2f}")            # Left aligned
print(f"Price: ${price:>10.2f}")            # Right aligned

# INPUT FROM USER (commented out so file runs automatically)
# The input() function always returns a string (text)

print("\nInput examples (commented out):")
print("# user_name = input('What is your name? ')")
print("# print(f'Hello, {user_name}!')")
print("# ")
print("# age_text = input('How old are you? ')")
print("# age_number = int(age_text)  # Convert to number")
print("# print(f'Next year you will be {age_number + 1}')")

# Simulating user input for demonstration:
simulated_name = "Alice"
simulated_age = "25"

print(f"\nSimulating input:")
print(f"User entered name: '{simulated_name}'")
print(f"User entered age: '{simulated_age}' (this is text)")
print(f"Converting age to number: {int(simulated_age)} (this is a number)")
print(f"Next year they'll be: {int(simulated_age) + 1}")

# =============================================================================
# 8. PUTTING IT ALL TOGETHER - COMPREHENSIVE EXAMPLES
# =============================================================================

print("\n🎯 SECTION 8: COMPREHENSIVE EXAMPLES")
print("-" * 50)

# EXAMPLE 1: Student Grade Calculator
print("\n📊 Example 1: Student Grade Calculator")
print("-" * 30)

# Student data
student_info = {
    "name": "Jordan Smith",
    "student_id": "JS2024",
    "grades": [88, 92, 85, 90, 87],
    "credits": [3, 4, 3, 3, 2]  # Credit hours for each course
}

# Calculate statistics
grades = student_info["grades"]
credits = student_info["credits"]

# Simple average
simple_average = sum(grades) / len(grades)

# Weighted average (considering credit hours)
total_points = sum(grade * credit for grade, credit in zip(grades, credits))
total_credits = sum(credits)
weighted_average = total_points / total_credits

# Determine letter grade
if weighted_average >= 93:
    letter_grade = "A"
elif weighted_average >= 90:
    letter_grade = "A-"
elif weighted_average >= 87:
    letter_grade = "B+"
elif weighted_average >= 83:
    letter_grade = "B"
elif weighted_average >= 80:
    letter_grade = "B-"
else:
    letter_grade = "C or below"

# Display results
print(f"Student: {student_info['name']} ({student_info['student_id']})")
print(f"Grades: {grades}")
print(f"Credits: {credits}")
print(f"Simple Average: {simple_average:.2f}")
print(f"Weighted Average: {weighted_average:.2f}")
print(f"Letter Grade: {letter_grade}")

# EXAMPLE 2: Shopping Cart Calculator
print("\n🛒 Example 2: Shopping Cart Calculator")
print("-" * 30)

# Shopping cart items
cart = [
    {"name": "Laptop", "price": 999.99, "quantity": 1, "taxable": True},
    {"name": "Mouse", "price": 29.99, "quantity": 2, "taxable": True},
    {"name": "Book", "price": 19.99, "quantity": 3, "taxable": False},  # Books not taxed
    {"name": "Headphones", "price": 79.99, "quantity": 1, "taxable": True}
]

# Calculate totals
subtotal = 0
tax_rate = 0.08  # 8% sales tax

print("SHOPPING CART:")
print("-" * 40)
for item in cart:
    item_total = item["price"] * item["quantity"]
    subtotal += item_total
    print(f"{item['name']:12} ${item['price']:7.2f} x {item['quantity']} = ${item_total:8.2f}")

# Calculate tax (only on taxable items)
tax_amount = 0
for item in cart:
    if item["taxable"]:
        item_total = item["price"] * item["quantity"]
        tax_amount += item_total * tax_rate

total = subtotal + tax_amount

print("-" * 40)
print(f"{'Subtotal:':20} ${subtotal:8.2f}")
print(f"{'Tax (8%):':20} ${tax_amount:8.2f}")
print(f"{'Total:':20} ${total:8.2f}")

# EXAMPLE 3: Text Analysis Tool
print("\n📝 Example 3: Text Analysis Tool")
print("-" * 30)

text = """Python is an amazing programming language. 
It's easy to learn and very powerful. 
Python is used for web development, data science, and automation."""

# Clean and prepare text
text_clean = text.lower().replace('\n', ' ')  # Remove newlines, make lowercase
words = text_clean.split()  # Split into words

# Remove punctuation from words
clean_words = []
for word in words:
    clean_word = word.strip('.,!?;:"()[]')  # Remove common punctuation
    if clean_word:  # Only add non-empty words
        clean_words.append(clean_word)

# Analysis
total_chars = len(text)
total_words = len(clean_words)
unique_words = len(set(clean_words))  # set() removes duplicates
avg_word_length = sum(len(word) for word in clean_words) / total_words

# Find most common words
word_count = {}
for word in clean_words:
    word_count[word] = word_count.get(word, 0) + 1

# Sort by frequency
most_common = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

print(f"Original text: {text}")
print(f"\nAnalysis Results:")
print(f"Total characters: {total_chars}")
print(f"Total words: {total_words}")
print(f"Unique words: {unique_words}")
print(f"Average word length: {avg_word_length:.1f} characters")
print(f"\nMost common words:")
for word, count in most_common[:5]:  # Top 5
    print(f"  '{word}': {count} times")

# =============================================================================
# 9. COMMON MISTAKES AND HOW TO AVOID THEM
# =============================================================================

print("\n⚠️ SECTION 9: COMMON MISTAKES TO AVOID")
print("-" * 50)

print("1. FORGETTING QUOTES AROUND STRINGS")
print("   ❌ name = Alice      # This will cause an error!")
print("   ✅ name = 'Alice'    # Correct way")

print("\n2. MIXING UP VARIABLE NAMES")
print("   ❌ student_name = 'John'")
print("       print(student_nam)  # Typo! Will cause error")
print("   ✅ student_name = 'John'")
print("       print(student_name)  # Correct spelling")

print("\n3. FORGETTING LIST INDEX STARTS AT 0")
print("   fruits = ['apple', 'banana', 'orange']")
print("   ❌ first_fruit = fruits[1]  # This gets 'banana', not 'apple'!")
print("   ✅ first_fruit = fruits[0]  # This gets 'apple'")

print("\n4. TRYING TO ADD DIFFERENT TYPES")
print("   ❌ result = '5' + 3      # Can't add string and number!")
print("   ✅ result = int('5') + 3 # Convert string to number first")
print("   ✅ result = '5' + '3'    # Or keep both as strings")

print("\n5. FORGETTING TO CONVERT INPUT TO NUMBERS")
print("   ❌ age = input('Age: ')  # This is always text!")
print("       next_year = age + 1  # Error: can't add 1 to text")
print("   ✅ age = int(input('Age: '))  # Convert to number")
print("       next_year = age + 1       # Now this works!")

# =============================================================================
# 10. PRACTICE EXERCISES WITH DETAILED SOLUTIONS
# =============================================================================

print("\n🏋️ SECTION 10: PRACTICE EXERCISES")
print("-" * 50)

print("\n📝 Exercise 1: Personal Information Manager")
print("-" * 30)

# Create a personal profile
my_profile = {
    "first_name": "Taylor",
    "last_name": "Johnson",
    "age": 22,
    "city": "Seattle",
    "hobbies": ["reading", "hiking", "cooking"],
    "favorite_numbers": [7, 13, 42],
    "is_student": True
}

# Display information in a nice format
full_name = f"{my_profile['first_name']} {my_profile['last_name']}"
hobbies_text = ", ".join(my_profile["hobbies"])
numbers_sum = sum(my_profile["favorite_numbers"])

print("PERSONAL PROFILE")
print("=" * 20)
print(f"Name: {full_name}")
print(f"Age: {my_profile['age']} years old")
print(f"Location: {my_profile['city']}")
print(f"Student Status: {'Yes' if my_profile['is_student'] else 'No'}")
print(f"Hobbies: {hobbies_text}")
print(f"Favorite Numbers: {my_profile['favorite_numbers']}")
print(f"Sum of Favorite Numbers: {numbers_sum}")

print("\n💰 Exercise 2: Budget Tracker")
print("-" * 30)

# Monthly budget data
budget = {
    "income": 3500.00,
    "expenses": {
        "rent": 1200.00,
        "food": 400.00,
        "transportation": 200.00,
        "utilities": 150.00,
        "entertainment": 300.00,
        "savings": 500.00
    }
}

# Calculate budget analysis
total_expenses = sum(budget["expenses"].values())
remaining_money = budget["income"] - total_expenses

print("MONTHLY BUDGET ANALYSIS")
print("=" * 30)
print(f"Income: ${budget['income']:,.2f}")
print("\nExpenses:")
for category, amount in budget["expenses"].items():
    percentage = (amount / budget["income"]) * 100
    print(f"  {category.title():15} ${amount:7.2f} ({percentage:4.1f}%)")

print("-" * 30)
print(f"Total Expenses: ${total_expenses:,.2f}")
print(f"Remaining: ${remaining_money:,.2f}")

if remaining_money > 0:
    print("✅ Budget is balanced!")
else:
    print("⚠️ Over budget!")

print("\n📚 Exercise 3: Library Book Manager")
print("-" * 30)

# Library books database
books = [
    {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960, "pages": 376, "available": True},
    {"title": "1984", "author": "George Orwell", "year": 1949, "pages": 328, "available": False},
    {"title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813, "pages": 432, "available": True},
    {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925, "pages": 180, "available": True}
]

# Library statistics
total_books = len(books)
available_books = sum(1 for book in books if book["available"])
total_pages = sum(book["pages"] for book in books)
avg_pages = total_pages / total_books

# Find oldest and newest books
oldest_book = min(books, key=lambda x: x["year"])
newest_book = max(books, key=lambda x: x["year"])

print("LIBRARY BOOK COLLECTION")
print("=" * 40)
print(f"Total Books: {total_books}")
print(f"Available: {available_books}")
print(f"Checked Out: {total_books - available_books}")
print(f"Average Pages: {avg_pages:.0f}")
print(f"Oldest Book: {oldest_book['title']} ({oldest_book['year']})")
print(f"Newest Book: {newest_book['title']} ({newest_book['year']})")

print("\nAVAILABLE BOOKS:")
for book in books:
    if book["available"]:
        print(f"  📖 {book['title']} by {book['author']} ({book['pages']} pages)")

# =============================================================================
# 11. SUMMARY AND NEXT STEPS
# =============================================================================

print("\n🎉 CONGRATULATIONS! You've completed Python Basics (Detailed)")
print("=" * 60)

concepts_covered = [
    "✅ Variables and how to name them properly",
    "✅ Data types: strings, integers, floats, booleans",
    "✅ String operations and formatting",
    "✅ Math operations and number conversions",
    "✅ Lists: creating, accessing, modifying",
    "✅ Dictionaries: key-value storage and retrieval",
    "✅ Input and output formatting",
    "✅ Common mistakes and how to avoid them",
    "✅ Real-world programming examples"
]

print("\n🏆 CONCEPTS YOU'VE MASTERED:")
for concept in concepts_covered:
    print(f"  {concept}")

print("\n🎯 WHAT YOU CAN DO NOW:")
abilities = [
    "• Store and manipulate different types of data",
    "• Create and work with collections (lists and dictionaries)",
    "• Perform calculations and format output nicely",
    "• Build simple data analysis programs",
    "• Debug common programming errors",
    "• Write clear, readable code with good variable names"
]

for ability in abilities:
    print(f"  {ability}")

print("\n🚀 READY FOR NEXT LEVEL:")
print("  You're now ready to move on to:")
print("  📁 02_control_structures.py - Learn if/else statements and loops")
print("  📁 Continue through the numbered files in order")

print("\n💡 KEEP PRACTICING:")
print("  • Try modifying the examples in this file")
print("  • Create your own variables and data structures")
print("  • Experiment with different combinations")
print("  • Don't worry about making mistakes - that's how you learn!")

print("\n🐍 Happy coding! You're doing great!")
