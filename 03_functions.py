# Functions - Reusable Code Blocks
# Functions are the building blocks of good Python code!

# =============================================================================
# 1. BASIC FUNCTIONS
# =============================================================================

# Simple function
def greet():
    print("Hello, World!")

# Call the function
greet()

# Function with parameters
def greet_person(name):
    print(f"Hello, {name}!")

greet_person("Alice")
greet_person("Bob")

# Function with multiple parameters
def introduce(name, age, city):
    print(f"Hi, I'm {name}. I'm {age} years old and I live in {city}.")

introduce("Charlie", 25, "New York")

# =============================================================================
# 2. RETURN VALUES
# =============================================================================

# Function that returns a value
def add_numbers(a, b):
    result = a + b
    return result

sum_result = add_numbers(5, 3)
print(f"5 + 3 = {sum_result}")

# Function with multiple return values
def get_name_parts(full_name):
    parts = full_name.split()
    first_name = parts[0]
    last_name = parts[-1]
    return first_name, last_name

first, last = get_name_parts("John Doe Smith")
print(f"First: {first}, Last: {last}")

# =============================================================================
# 3. DEFAULT PARAMETERS
# =============================================================================

def greet_with_title(name, title="Mr./Ms."):
    print(f"Hello, {title} {name}!")

greet_with_title("Johnson")           # Uses default title
greet_with_title("Smith", "Dr.")      # Uses custom title

def create_profile(name, age=25, city="Unknown"):
    return {
        "name": name,
        "age": age,
        "city": city
    }

profile1 = create_profile("Alice")
profile2 = create_profile("Bob", 30, "Boston")
print(profile1)
print(profile2)

# =============================================================================
# 4. *ARGS AND **KWARGS (ADVANCED BUT USEFUL)
# =============================================================================

# *args - variable number of arguments
def calculate_average(*numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

avg1 = calculate_average(10, 20, 30)
avg2 = calculate_average(1, 2, 3, 4, 5)
print(f"Average 1: {avg1}")
print(f"Average 2: {avg2}")

# **kwargs - variable number of keyword arguments
def create_user(**details):
    print("Creating user with details:")
    for key, value in details.items():
        print(f"  {key}: {value}")

create_user(name="Alice", age=30, email="alice@email.com", city="New York")

# =============================================================================
# 5. LAMBDA FUNCTIONS (ANONYMOUS FUNCTIONS)
# =============================================================================

# Regular function
def square(x):
    return x ** 2

# Lambda function (same thing, shorter)
square_lambda = lambda x: x ** 2

print(f"Square of 5: {square(5)}")
print(f"Square of 5 (lambda): {square_lambda(5)}")

# Lambda with multiple parameters
multiply = lambda x, y: x * y
print(f"3 * 4 = {multiply(3, 4)}")

# Lambda functions are great with map, filter, sort
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

print(f"Original: {numbers}")
print(f"Squared: {squared}")
print(f"Evens: {evens}")

# =============================================================================
# 6. SCOPE - WHERE VARIABLES LIVE
# =============================================================================

global_var = "I'm global!"

def scope_example():
    local_var = "I'm local!"
    print(f"Inside function: {global_var}")
    print(f"Inside function: {local_var}")

scope_example()
print(f"Outside function: {global_var}")
# print(local_var)  # This would cause an error!

# =============================================================================
# 7. USEFUL BUILT-IN FUNCTIONS
# =============================================================================

# len() - length
my_list = [1, 2, 3, 4, 5]
print(f"Length: {len(my_list)}")

# min() and max()
print(f"Min: {min(my_list)}")
print(f"Max: {max(my_list)}")

# sum()
print(f"Sum: {sum(my_list)}")

# sorted()
unsorted = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_list = sorted(unsorted)
print(f"Sorted: {sorted_list}")

# zip() - combine lists
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))
print(f"Combined: {combined}")

# enumerate() - get index and value
fruits = ["apple", "banana", "orange"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# =============================================================================
# 8. PRACTICAL EXERCISES
# =============================================================================

print("\n" + "="*50)
print("PRACTICE TIME!")
print("="*50)

# Exercise 1: Temperature converter
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

print(f"25°C = {celsius_to_fahrenheit(25):.1f}°F")
print(f"77°F = {fahrenheit_to_celsius(77):.1f}°C")

# Exercise 2: Password validator
def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and digits"
    
    return True, "Password is valid!"

# Test passwords
test_passwords = ["weak", "StrongPass123", "nodigits", "NOLOWER123"]
for pwd in test_passwords:
    is_valid, message = validate_password(pwd)
    print(f"'{pwd}': {message}")

# Exercise 3: List statistics
def list_stats(numbers):
    if not numbers:
        return {"error": "Empty list"}
    
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

test_numbers = [10, 20, 30, 40, 50]
stats = list_stats(test_numbers)
print(f"\nStats for {test_numbers}:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Exercise 4: Word frequency counter
def count_words(text):
    words = text.lower().split()
    word_count = {}
    for word in words:
        # Remove punctuation
        clean_word = word.strip(".,!?;:")
        word_count[clean_word] = word_count.get(clean_word, 0) + 1
    return word_count

sample_text = "Python is great! Python is powerful. I love Python programming."
word_freq = count_words(sample_text)
print(f"\nWord frequency in: '{sample_text}'")
for word, count in word_freq.items():
    print(f"  {word}: {count}")

print("\n🎉 Amazing! You've mastered functions!")
print("Next: Run this file and then move to 04_classes.py")
