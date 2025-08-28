# Control Structures - Making Decisions and Loops
# This is where Python gets really powerful!

# =============================================================================
# 1. IF/ELSE STATEMENTS - MAKING DECISIONS
# =============================================================================

age = 18

# Basic if/else
if age >= 18:
    print("You are an adult!")
else:
    print("You are a minor.")

# Multiple conditions with elif
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")

# Logical operators
temperature = 75
weather = "sunny"

if temperature > 70 and weather == "sunny":
    print("Perfect day for a picnic!")
elif temperature > 70 or weather == "sunny":
    print("Pretty good day!")
else:
    print("Maybe stay inside today.")

# =============================================================================
# 2. FOR LOOPS - REPEATING CODE
# =============================================================================

# Loop through a list
fruits = ["apple", "banana", "orange", "grape"]
print("My fruits:")
for fruit in fruits:
    print(f"  - {fruit}")

# Loop through a string
word = "Python"
print("\nLetters in 'Python':")
for letter in word:
    print(letter)

# Loop with range() - SUPER USEFUL!
print("\nCounting to 5:")
for i in range(5):          # 0, 1, 2, 3, 4
    print(f"Count: {i}")

print("\nCounting 1 to 10:")
for i in range(1, 11):      # 1, 2, 3, ..., 10
    print(f"Number: {i}")

print("\nEven numbers 0-10:")
for i in range(0, 11, 2):   # 0, 2, 4, 6, 8, 10
    print(i)

# Loop through dictionary
person = {"name": "Bob", "age": 25, "city": "Boston"}
print("\nPerson details:")
for key, value in person.items():
    print(f"{key}: {value}")

# =============================================================================
# 3. WHILE LOOPS - REPEAT UNTIL CONDITION IS FALSE
# =============================================================================

# Basic while loop
count = 0
print("\nWhile loop countdown:")
while count < 5:
    print(f"Count is: {count}")
    count += 1  # Same as count = count + 1

# While loop with user input (commented out for automatic running)
"""
password = ""
while password != "secret":
    password = input("Enter password: ")
    if password != "secret":
        print("Wrong password, try again!")
print("Access granted!")
"""

# =============================================================================
# 4. LIST COMPREHENSIONS - PYTHON'S SUPERPOWER!
# =============================================================================

# Traditional way
squares = []
for i in range(10):
    squares.append(i ** 2)
print(f"Squares (traditional): {squares}")

# List comprehension way (MUCH SHORTER!)
squares_comp = [i ** 2 for i in range(10)]
print(f"Squares (comprehension): {squares_comp}")

# With conditions
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
print(f"Even squares: {even_squares}")

# Working with strings
words = ["python", "java", "javascript", "go"]
long_words = [word.upper() for word in words if len(word) > 4]
print(f"Long words (uppercase): {long_words}")

# =============================================================================
# 5. BREAK AND CONTINUE
# =============================================================================

print("\nBreak example (stop when we find 'banana'):")
fruits = ["apple", "banana", "orange", "grape"]
for fruit in fruits:
    if fruit == "banana":
        print(f"Found {fruit}! Stopping here.")
        break
    print(f"Checking: {fruit}")

print("\nContinue example (skip 'banana'):")
for fruit in fruits:
    if fruit == "banana":
        print(f"Skipping {fruit}")
        continue
    print(f"Processing: {fruit}")

# =============================================================================
# 6. PRACTICAL EXERCISES
# =============================================================================

print("\n" + "="*50)
print("PRACTICE TIME!")
print("="*50)

# Exercise 1: FizzBuzz (classic programming challenge)
print("\nFizzBuzz (1-20):")
for i in range(1, 21):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

# Exercise 2: Find all even numbers in a list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [num for num in numbers if num % 2 == 0]
print(f"\nEven numbers: {even_numbers}")

# Exercise 3: Count vowels in a string
text = "Hello World"
vowels = "aeiouAEIOU"
vowel_count = 0
for char in text:
    if char in vowels:
        vowel_count += 1
print(f"\nVowels in '{text}': {vowel_count}")

# Exercise 4: Simple number guessing game logic
secret_number = 7
guess = 5

if guess == secret_number:
    print(f"\n🎉 Correct! The number was {secret_number}")
elif guess < secret_number:
    print(f"\n📈 Too low! The number was {secret_number}")
else:
    print(f"\n📉 Too high! The number was {secret_number}")

print("\n🎉 Excellent! You've mastered control structures!")
print("Next: Run this file and then move to 03_functions.py")
