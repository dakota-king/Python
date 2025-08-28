# Useful Python Libraries - Expand Your Toolkit!
# These libraries will make you incredibly productive in Python

import json
import datetime
import random
import math
import os
import sys
from collections import Counter, defaultdict
import re

# =============================================================================
# 1. DATETIME - WORKING WITH DATES AND TIMES
# =============================================================================

print("=== DateTime Library ===")

# Get current date and time
now = datetime.datetime.now()
today = datetime.date.today()
current_time = datetime.time(now.hour, now.minute, now.second)

print(f"Current datetime: {now}")
print(f"Today's date: {today}")
print(f"Current time: {current_time}")

# Formatting dates
print(f"Formatted date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Readable format: {now.strftime('%B %d, %Y at %I:%M %p')}")

# Date arithmetic
tomorrow = today + datetime.timedelta(days=1)
last_week = today - datetime.timedelta(weeks=1)
next_month = today + datetime.timedelta(days=30)

print(f"Tomorrow: {tomorrow}")
print(f"Last week: {last_week}")
print(f"Next month: {next_month}")

# Create specific dates
birthday = datetime.date(1990, 5, 15)
age_in_days = (today - birthday).days
print(f"If born on {birthday}, you'd be {age_in_days} days old!")

# =============================================================================
# 2. RANDOM - GENERATING RANDOM DATA
# =============================================================================

print(f"\n=== Random Library ===")

# Random numbers
print(f"Random float 0-1: {random.random()}")
print(f"Random int 1-10: {random.randint(1, 10)}")
print(f"Random float 1-100: {random.uniform(1, 100):.2f}")

# Random choices
colors = ["red", "green", "blue", "yellow", "purple"]
print(f"Random color: {random.choice(colors)}")
print(f"3 random colors: {random.choices(colors, k=3)}")

# Shuffle and sample
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
shuffled = numbers.copy()
random.shuffle(shuffled)
print(f"Original: {numbers}")
print(f"Shuffled: {shuffled}")
print(f"Random sample of 3: {random.sample(numbers, 3)}")

# Random string generation
def generate_password(length=8):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))

print(f"Random password: {generate_password(12)}")

# =============================================================================
# 3. MATH - MATHEMATICAL FUNCTIONS
# =============================================================================

print(f"\n=== Math Library ===")

# Basic math functions
print(f"Square root of 16: {math.sqrt(16)}")
print(f"2 to the power of 3: {math.pow(2, 3)}")
print(f"Ceiling of 4.3: {math.ceil(4.3)}")
print(f"Floor of 4.7: {math.floor(4.7)}")
print(f"Absolute value of -5: {math.fabs(-5)}")

# Trigonometry (angles in radians)
angle = math.pi / 4  # 45 degrees
print(f"Sin(45°): {math.sin(angle):.3f}")
print(f"Cos(45°): {math.cos(angle):.3f}")

# Logarithms
print(f"Natural log of 10: {math.log(10):.3f}")
print(f"Log base 10 of 100: {math.log10(100)}")

# Constants
print(f"Pi: {math.pi:.6f}")
print(f"Euler's number: {math.e:.6f}")

# =============================================================================
# 4. COLLECTIONS - SPECIALIZED DATA STRUCTURES
# =============================================================================

print(f"\n=== Collections Library ===")

# Counter - count elements
text = "hello world"
letter_count = Counter(text)
print(f"Letter frequency in '{text}': {letter_count}")
print(f"Most common letters: {letter_count.most_common(3)}")

# Count words in a sentence
sentence = "the quick brown fox jumps over the lazy dog the fox"
word_count = Counter(sentence.split())
print(f"Word frequency: {word_count}")

# defaultdict - dictionary with default values
from collections import defaultdict

# Group words by their first letter
words = ["apple", "banana", "cherry", "avocado", "blueberry", "apricot"]
grouped = defaultdict(list)

for word in words:
    grouped[word[0]].append(word)

print(f"Words grouped by first letter: {dict(grouped)}")

# =============================================================================
# 5. REGULAR EXPRESSIONS (RE) - PATTERN MATCHING
# =============================================================================

print(f"\n=== Regular Expressions ===")

# Basic pattern matching
text = "My phone number is 123-456-7890 and my email is john@example.com"

# Find phone numbers
phone_pattern = r'\d{3}-\d{3}-\d{4}'
phone_match = re.search(phone_pattern, text)
if phone_match:
    print(f"Found phone: {phone_match.group()}")

# Find email addresses
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
email_match = re.search(email_pattern, text)
if email_match:
    print(f"Found email: {email_match.group()}")

# Find all numbers
numbers_text = "I have 5 apples, 10 oranges, and 3 bananas"
numbers = re.findall(r'\d+', numbers_text)
print(f"All numbers found: {numbers}")

# Replace patterns
messy_text = "Hello    world!   How   are    you?"
clean_text = re.sub(r'\s+', ' ', messy_text)  # Replace multiple spaces with single space
print(f"Original: '{messy_text}'")
print(f"Cleaned:  '{clean_text}'")

# Validate input
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

test_emails = ["valid@example.com", "invalid-email", "test@test.co.uk"]
for email in test_emails:
    print(f"{email}: {'Valid' if validate_email(email) else 'Invalid'}")

# =============================================================================
# 6. OS - OPERATING SYSTEM INTERFACE
# =============================================================================

print(f"\n=== OS Library ===")

# Get system information
print(f"Operating system: {os.name}")
print(f"Current directory: {os.getcwd()}")
print(f"User home directory: {os.path.expanduser('~')}")

# Environment variables
print(f"PATH variable length: {len(os.environ.get('PATH', ''))}")
print(f"Python path: {sys.executable}")

# File and directory operations
current_files = [f for f in os.listdir('.') if f.endswith('.py')]
print(f"Python files in current directory: {current_files}")

# Path operations
example_path = "/home/user/documents/file.txt"
print(f"Directory: {os.path.dirname(example_path)}")
print(f"Filename: {os.path.basename(example_path)}")
print(f"File extension: {os.path.splitext(example_path)[1]}")

# =============================================================================
# 7. JSON - WORKING WITH JSON DATA
# =============================================================================

print(f"\n=== JSON Library ===")

# Python data to JSON
python_data = {
    "name": "Alice",
    "age": 30,
    "skills": ["Python", "JavaScript", "SQL"],
    "is_student": False,
    "graduation_date": None
}

json_string = json.dumps(python_data, indent=2)
print("Python data as JSON:")
print(json_string)

# JSON to Python data
json_data = '{"temperature": 25.5, "humidity": 60, "location": "New York"}'
python_dict = json.loads(json_data)
print(f"\nJSON to Python: {python_dict}")
print(f"Temperature: {python_dict['temperature']}°C")

# =============================================================================
# 8. PRACTICAL EXAMPLES AND MINI-PROJECTS
# =============================================================================

print(f"\n" + "="*50)
print("PRACTICAL EXAMPLES!")
print("="*50)

# Example 1: Password Generator with Requirements
def generate_secure_password(length=12, include_symbols=True):
    """Generate a secure password with specific requirements."""
    import string
    
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""
    
    # Ensure password has at least one of each type
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits)
    ]
    
    if include_symbols:
        password.append(random.choice(symbols))
    
    # Fill the rest with random characters
    all_chars = lowercase + uppercase + digits + symbols
    for _ in range(length - len(password)):
        password.append(random.choice(all_chars))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)

print("Secure password examples:")
for i in range(3):
    print(f"  {i+1}. {generate_secure_password(16)}")

# Example 2: Log File Analyzer
def analyze_log_entries():
    """Simulate analyzing log entries with datetime and regex."""
    
    # Simulate log entries
    log_entries = [
        "2024-01-15 10:30:15 INFO User login successful for user123",
        "2024-01-15 10:31:22 ERROR Database connection failed",
        "2024-01-15 10:32:10 INFO User logout for user123",
        "2024-01-15 10:33:45 WARNING High memory usage detected",
        "2024-01-15 10:34:12 ERROR Failed to send email notification",
        "2024-01-15 10:35:01 INFO System backup completed"
    ]
    
    # Parse log entries
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)'
    
    parsed_logs = []
    for entry in log_entries:
        match = re.match(log_pattern, entry)
        if match:
            timestamp_str, level, message = match.groups()
            timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            parsed_logs.append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
    
    # Analyze logs
    level_count = Counter(log['level'] for log in parsed_logs)
    
    print("\n=== Log Analysis ===")
    print(f"Total log entries: {len(parsed_logs)}")
    print("Log levels:")
    for level, count in level_count.items():
        print(f"  {level}: {count}")
    
    # Find errors
    errors = [log for log in parsed_logs if log['level'] == 'ERROR']
    if errors:
        print("\nError messages:")
        for error in errors:
            print(f"  {error['timestamp'].strftime('%H:%M:%S')}: {error['message']}")

analyze_log_entries()

# Example 3: Simple Data Validator
class DataValidator:
    """A class to validate different types of data using various libraries."""
    
    @staticmethod
    def validate_date(date_string, format_string='%Y-%m-%d'):
        """Validate if string is a valid date."""
        try:
            datetime.datetime.strptime(date_string, format_string)
            return True, "Valid date"
        except ValueError:
            return False, "Invalid date format"
    
    @staticmethod
    def validate_number_range(value, min_val=None, max_val=None):
        """Validate if number is within specified range."""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and num > max_val:
                return False, f"Value must be <= {max_val}"
            return True, "Valid number"
        except ValueError:
            return False, "Not a valid number"
    
    @staticmethod
    def validate_password_strength(password):
        """Check password strength using regex."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        
        checks = {
            'uppercase': re.search(r'[A-Z]', password),
            'lowercase': re.search(r'[a-z]', password),
            'digit': re.search(r'\d', password),
            'special': re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
        }
        
        missing = [check for check, found in checks.items() if not found]
        
        if missing:
            return False, f"Password missing: {', '.join(missing)}"
        return True, "Strong password"

# Test the validator
validator = DataValidator()

test_data = [
    ("2024-01-15", "date"),
    ("2024-13-40", "date"),
    ("25.5", "number"),
    ("abc", "number"),
    ("StrongPass123!", "password"),
    ("weak", "password")
]

print("\n=== Data Validation Tests ===")
for data, data_type in test_data:
    if data_type == "date":
        valid, message = validator.validate_date(data)
    elif data_type == "number":
        valid, message = validator.validate_number_range(data, 0, 100)
    elif data_type == "password":
        valid, message = validator.validate_password_strength(data)
    
    status = "✅" if valid else "❌"
    print(f"{status} {data} ({data_type}): {message}")

# Example 4: Simple Statistics Calculator
def calculate_statistics(numbers):
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"error": "Empty list"}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    # Basic stats
    total = sum(numbers)
    mean = total / n
    
    # Median
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    
    # Standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = math.sqrt(variance)
    
    return {
        "count": n,
        "sum": total,
        "mean": round(mean, 2),
        "median": median,
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
        "std_dev": round(std_dev, 2)
    }

# Generate random test data
test_numbers = [random.randint(1, 100) for _ in range(20)]
stats = calculate_statistics(test_numbers)

print(f"\n=== Statistics for Random Numbers ===")
print(f"Numbers: {sorted(test_numbers)}")
for key, value in stats.items():
    print(f"{key.title()}: {value}")

print(f"\n🎉 Outstanding! You've learned essential Python libraries!")
print("You now have a solid foundation in Python programming!")
print("\nNext steps:")
print("1. Practice with real projects")
print("2. Learn web frameworks (Flask/Django)")
print("3. Explore data science (pandas, numpy, matplotlib)")
print("4. Try automation (selenium, requests)")
print("5. Build something awesome! 🚀")
