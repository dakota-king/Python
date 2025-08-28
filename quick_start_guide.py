# 🚀 PYTHON QUICK START GUIDE - Learn Fast!
# This is your roadmap to learning Python as quickly as possible

print("🐍 Welcome to Python Quick Start Guide!")
print("="*50)

# =============================================================================
# HOW TO RUN FILES IN THIS SYSTEM
# =============================================================================

print("\n🚀 HOW TO RUN FILES:")
print("="*50)

run_instructions = [
    "1. Open terminal/command prompt",
    "2. Navigate to the LearnPython folder: cd LearnPython",
    "3. Run any file: python filename.py",
    "",
    "Examples:",
    "  python quick_start_guide.py     (this file)",
    "  python 01_python_basics.py      (start learning)",
    "  python 08_advanced_python_concepts.py  (interview prep)",
    "",
    "💡 Tips:",
    "  • If 'python' doesn't work, try 'python3' or 'py'",
    "  • Each file is self-contained and runs independently",
    "  • Files show output immediately when run",
    "  • Advanced files (08-10) may take 30-60 seconds to complete",
    "",
    "🎯 Recommended order:",
    "  Complete Beginner: 00 → 01_detailed → 01-07 → 08-10",
    "  Some Experience: 01-07 → 08-10",
    "  Interview Prep: Review 01-07 → Focus on 08-10"
]

for instruction in run_instructions:
    print(f"   {instruction}")

print("\n" + "="*50)

# =============================================================================
# FASTEST PATH TO PYTHON MASTERY
# =============================================================================

learning_path = {
    "Day 1-2": [
        "✅ Variables and data types (strings, numbers, lists, dictionaries)",
        "✅ Basic operations and string formatting",
        "✅ Input/output with print() and input()",
        "📁 Complete Beginner: 00_absolute_beginner_start.py",
        "📁 Detailed Version: 01_python_basics_detailed.py",
        "📁 Standard Version: 01_python_basics.py"
    ],
    
    "Day 3-4": [
        "✅ If/else statements and logical operators",
        "✅ For loops and while loops",
        "✅ List comprehensions (Python's superpower!)",
        "✅ Break and continue statements",
        "📁 Practice with: 02_control_structures.py"
    ],
    
    "Day 5-6": [
        "✅ Writing functions with parameters and return values",
        "✅ Lambda functions and built-in functions",
        "✅ *args and **kwargs",
        "✅ Scope and variable visibility",
        "📁 Practice with: 03_functions.py"
    ],
    
    "Day 7-8": [
        "✅ Classes and objects (OOP basics)",
        "✅ Inheritance and method overriding",
        "✅ Special methods (__init__, __str__, etc.)",
        "📁 Practice with: 04_classes.py"
    ],
    
    "Day 9-10": [
        "✅ File reading and writing",
        "✅ JSON data handling",
        "✅ Error handling with try/except",
        "✅ Working with CSV files",
        "📁 Practice with: 05_file_handling.py"
    ],
    
    "Day 11-12": [
        "✅ Essential libraries (datetime, random, math, os)",
        "✅ Regular expressions for text processing",
        "✅ Collections module (Counter, defaultdict)",
        "📁 Practice with: 06_useful_libraries.py"
    ],
    
    "Day 13-14": [
        "✅ Build real projects to solidify knowledge",
        "✅ Number guessing game, todo manager, calculator",
        "✅ Password manager and expense tracker",
        "📁 Practice with: 07_mini_projects.py"
    ]
}

# Display the learning path
for timeframe, tasks in learning_path.items():
    print(f"\n📅 {timeframe}:")
    for task in tasks:
        print(f"   {task}")

# =============================================================================
# ESSENTIAL PYTHON CONCEPTS - CHEAT SHEET
# =============================================================================

print(f"\n\n📚 PYTHON ESSENTIALS CHEAT SHEET")
print("="*50)

cheat_sheet = {
    "Variables": {
        "code": '''name = "Alice"
age = 25
is_student = True''',
        "tip": "No need to declare types - Python figures it out!"
    },
    
    "Lists": {
        "code": '''fruits = ["apple", "banana", "orange"]
fruits.append("grape")          # Add item
fruits[0]                       # First item: "apple"
fruits[-1]                      # Last item: "grape"
fruits[1:3]                     # Slice: ["banana", "orange"]''',
        "tip": "Lists are ordered, changeable, and allow duplicates"
    },
    
    "Dictionaries": {
        "code": '''person = {"name": "Bob", "age": 30}
person["name"]                  # Access value: "Bob"
person["email"] = "bob@email.com"  # Add new key-value
person.keys()                   # All keys
person.values()                 # All values''',
        "tip": "Dictionaries store key-value pairs, like JSON objects"
    },
    
    "If Statements": {
        "code": '''if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")''',
        "tip": "Use elif for multiple conditions, else for default case"
    },
    
    "Loops": {
        "code": '''# For loop
for fruit in fruits:
    print(fruit)

# Range loop
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

# While loop
count = 0
while count < 3:
    print(count)
    count += 1''',
        "tip": "For loops are for iterating, while loops for conditions"
    },
    
    "Functions": {
        "code": '''def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

result = greet("Alice")         # "Hello, Alice!"
result = greet("Bob", "Hi")     # "Hi, Bob!"''',
        "tip": "Functions make code reusable and organized"
    },
    
    "List Comprehensions": {
        "code": '''# Traditional way
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension (Pythonic way!)
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]''',
        "tip": "List comprehensions are faster and more readable"
    },
    
    "Error Handling": {
        "code": '''try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("This always runs")''',
        "tip": "Always handle errors gracefully in production code"
    },
    
    "File Operations": {
        "code": '''# Reading a file
with open("file.txt", "r") as f:
    content = f.read()

# Writing a file
with open("file.txt", "w") as f:
    f.write("Hello, World!")

# JSON handling
import json
data = {"name": "Alice", "age": 25}
json_string = json.dumps(data)
parsed_data = json.loads(json_string)''',
        "tip": "Always use 'with' statement for file operations"
    },
    
    "Classes": {
        "code": '''class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says Woof!"

my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.bark())            # "Buddy says Woof!"''',
        "tip": "Classes help organize related data and functions"
    }
}

# Display cheat sheet
for concept, details in cheat_sheet.items():
    print(f"\n🔹 {concept.upper()}:")
    print(f"💡 {details['tip']}")
    print("📝 Example:")
    for line in details['code'].split('\n'):
        if line.strip():
            print(f"   {line}")

# =============================================================================
# PRACTICE EXERCISES - DO THESE TO LEARN FAST!
# =============================================================================

print(f"\n\n🏋️ PRACTICE EXERCISES")
print("="*50)

exercises = [
    {
        "title": "Variables & Data Types",
        "task": "Create variables for your name, age, favorite foods (list), and a profile (dictionary)",
        "difficulty": "Beginner"
    },
    {
        "title": "String Manipulation",
        "task": "Take a sentence, count vowels, reverse words, and make it title case",
        "difficulty": "Beginner"
    },
    {
        "title": "List Operations",
        "task": "Create a list of numbers, find sum, average, max, min, and even numbers",
        "difficulty": "Beginner"
    },
    {
        "title": "Loops & Conditions",
        "task": "Write FizzBuzz: print numbers 1-100, but 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both",
        "difficulty": "Intermediate"
    },
    {
        "title": "Functions",
        "task": "Create a temperature converter (C to F and F to C) with input validation",
        "difficulty": "Intermediate"
    },
    {
        "title": "File Handling",
        "task": "Read a text file, count words, find most common word, save results to JSON",
        "difficulty": "Intermediate"
    },
    {
        "title": "Classes",
        "task": "Create a BankAccount class with deposit, withdraw, and balance methods",
        "difficulty": "Advanced"
    },
    {
        "title": "Mini Project",
        "task": "Build a contact book that saves/loads from JSON with add, search, delete functions",
        "difficulty": "Advanced"
    }
]

for i, exercise in enumerate(exercises, 1):
    difficulty_icon = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    icon = difficulty_icon.get(exercise["difficulty"], "⚪")
    
    print(f"\n{i}. {exercise['title']} {icon}")
    print(f"   📋 {exercise['task']}")
    print(f"   🎯 Level: {exercise['difficulty']}")

# =============================================================================
# RESOURCES AND NEXT STEPS
# =============================================================================

print(f"\n\n📖 LEARNING RESOURCES")
print("="*50)

resources = {
    "Official Documentation": "https://docs.python.org/3/",
    "Interactive Learning": "https://www.codecademy.com/learn/learn-python-3",
    "Practice Problems": "https://leetcode.com/problemset/all/ (filter by Python)",
    "Real Projects": "https://github.com/tuvtran/project-based-learning#python",
    "Python Community": "https://www.reddit.com/r/learnpython/",
    "Free Book": "https://automatetheboringstuff.com/",
    "Video Tutorials": "https://www.youtube.com/watch?v=_uQrJ0TkZlc (Python Crash Course)"
}

print("🌐 Online Resources:")
for resource, link in resources.items():
    print(f"   • {resource}: {link}")

# Next steps after basics
next_steps = [
    "🌐 Web Development: Learn Flask or Django for building web applications",
    "📊 Data Science: Master pandas, numpy, matplotlib for data analysis",
    "🤖 Machine Learning: Explore scikit-learn, TensorFlow for AI projects",
    "🔧 Automation: Use selenium, requests for web scraping and automation",
    "🎮 Game Development: Try pygame for creating simple games",
    "☁️ Cloud Computing: Learn AWS, Docker for deploying applications",
    "📱 GUI Apps: Build desktop apps with tkinter or PyQt",
    "🔗 APIs: Create REST APIs with FastAPI or Flask"
]

print(f"\n🚀 SPECIALIZATION PATHS:")
for step in next_steps:
    print(f"   {step}")

# Final motivation
print(f"\n\n🎯 FINAL TIPS FOR FAST LEARNING:")
motivation = [
    "💪 Code every day, even if just 15 minutes",
    "🛠️ Build projects, don't just read tutorials",
    "❓ Ask questions on Stack Overflow and Reddit",
    "👥 Join Python communities and Discord servers",
    "📚 Read other people's code to learn new techniques",
    "🐛 Don't fear errors - they're your best teachers!",
    "🎉 Celebrate small wins and progress",
    "🔄 Practice, practice, practice!"
]

for tip in motivation:
    print(f"   {tip}")

print(f"\n🐍 Remember: Python is designed to be readable and fun!")
print(f"   You've got this! Start with 01_python_basics.py and work your way through.")
print(f"   In just 2 weeks, you'll be writing real Python programs! 🚀✨")
