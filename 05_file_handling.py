# File Handling and Error Management
# Learn to work with files and handle errors gracefully!

import os
import json
from datetime import datetime

# =============================================================================
# 1. READING FILES
# =============================================================================

# Create a sample text file first
sample_text = """Welcome to Python file handling!
This is line 2.
This is line 3.
Python makes file handling easy!"""

with open("sample.txt", "w") as file:
    file.write(sample_text)

# Method 1: Read entire file
print("=== Reading entire file ===")
with open("sample.txt", "r") as file:
    content = file.read()
    print(content)

# Method 2: Read line by line
print("\n=== Reading line by line ===")
with open("sample.txt", "r") as file:
    for line_number, line in enumerate(file, 1):
        print(f"Line {line_number}: {line.strip()}")

# Method 3: Read all lines into a list
print("\n=== Reading all lines into list ===")
with open("sample.txt", "r") as file:
    lines = file.readlines()
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"  {i+1}: {line.strip()}")

# =============================================================================
# 2. WRITING FILES
# =============================================================================

# Writing text to a file
data_to_write = [
    "Python Programming Tips:",
    "1. Use meaningful variable names",
    "2. Write comments for complex code",
    "3. Follow PEP 8 style guidelines",
    "4. Handle errors gracefully"
]

# Write mode ('w') - overwrites existing file
with open("tips.txt", "w") as file:
    for tip in data_to_write:
        file.write(tip + "\n")

print("Created tips.txt file!")

# Append mode ('a') - adds to existing file
with open("tips.txt", "a") as file:
    file.write("5. Practice regularly!\n")
    file.write("6. Build projects to learn!\n")

print("Added more tips to the file!")

# Read and display the updated file
with open("tips.txt", "r") as file:
    print("\n=== Contents of tips.txt ===")
    print(file.read())

# =============================================================================
# 3. WORKING WITH JSON FILES
# =============================================================================

# Create some sample data
student_data = {
    "students": [
        {
            "id": 1,
            "name": "Alice Johnson",
            "age": 20,
            "courses": ["Python", "Data Science", "Web Development"],
            "gpa": 3.8
        },
        {
            "id": 2,
            "name": "Bob Smith",
            "age": 22,
            "courses": ["Python", "Machine Learning"],
            "gpa": 3.6
        },
        {
            "id": 3,
            "name": "Charlie Brown",
            "age": 21,
            "courses": ["Python", "Database Design", "Web Development"],
            "gpa": 3.9
        }
    ]
}

# Write JSON data to file
with open("students.json", "w") as file:
    json.dump(student_data, file, indent=2)

print("Created students.json file!")

# Read JSON data from file
with open("students.json", "r") as file:
    loaded_data = json.load(file)

print("\n=== Student Data from JSON ===")
for student in loaded_data["students"]:
    print(f"Name: {student['name']}")
    print(f"  Age: {student['age']}")
    print(f"  GPA: {student['gpa']}")
    print(f"  Courses: {', '.join(student['courses'])}")
    print()

# =============================================================================
# 4. ERROR HANDLING WITH TRY/EXCEPT
# =============================================================================

print("=== Error Handling Examples ===")

# Example 1: File not found
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("❌ Error: File not found!")
    print("✅ Created a new file instead.")
    with open("nonexistent.txt", "w") as file:
        file.write("This file was created because the original didn't exist!")

# Example 2: Division by zero
def safe_divide(a, b):
    try:
        result = a / b
        return f"{a} ÷ {b} = {result}"
    except ZeroDivisionError:
        return "❌ Error: Cannot divide by zero!"
    except TypeError:
        return "❌ Error: Please provide numbers only!"

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide("10", 2))

# Example 3: Multiple exceptions
def process_user_input(user_input):
    try:
        # Try to convert to integer
        number = int(user_input)
        
        # Try to access a list element
        my_list = [1, 2, 3, 4, 5]
        result = my_list[number]
        
        return f"Element at index {number}: {result}"
        
    except ValueError:
        return "❌ Error: Please enter a valid number!"
    except IndexError:
        return "❌ Error: Index out of range!"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Test the function
test_inputs = ["2", "10", "abc", "-1"]
for test_input in test_inputs:
    print(f"Input '{test_input}': {process_user_input(test_input)}")

# =============================================================================
# 5. FILE OPERATIONS WITH ERROR HANDLING
# =============================================================================

def safe_file_operations():
    """Demonstrate safe file operations with proper error handling."""
    
    # Create a log file with timestamp
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open("app.log", "a") as log_file:
                log_file.write(log_entry)
        except Exception as e:
            print(f"Failed to write to log: {e}")
    
    # Safe file reading function
    def read_config_file(filename):
        try:
            with open(filename, "r") as file:
                config = json.load(file)
                log_message(f"Successfully loaded config from {filename}")
                return config
        except FileNotFoundError:
            log_message(f"Config file {filename} not found, using defaults")
            return {"theme": "light", "language": "en", "debug": False}
        except json.JSONDecodeError:
            log_message(f"Invalid JSON in {filename}, using defaults")
            return {"theme": "light", "language": "en", "debug": False}
        except Exception as e:
            log_message(f"Unexpected error reading {filename}: {e}")
            return {}
    
    # Create a sample config file
    default_config = {
        "theme": "dark",
        "language": "en",
        "debug": True,
        "auto_save": True
    }
    
    with open("config.json", "w") as file:
        json.dump(default_config, file, indent=2)
    
    # Test the safe file reading
    config = read_config_file("config.json")
    print("\n=== Configuration Settings ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Test with non-existent file
    missing_config = read_config_file("missing_config.json")
    print("\n=== Default Configuration (file not found) ===")
    for key, value in missing_config.items():
        print(f"{key}: {value}")

safe_file_operations()

# =============================================================================
# 6. WORKING WITH CSV DATA
# =============================================================================

import csv

# Create sample CSV data
csv_data = [
    ["Name", "Age", "City", "Occupation"],
    ["Alice Johnson", "28", "New York", "Data Scientist"],
    ["Bob Smith", "32", "San Francisco", "Software Engineer"],
    ["Charlie Brown", "25", "Chicago", "Web Developer"],
    ["Diana Prince", "30", "Seattle", "Product Manager"]
]

# Write CSV file
with open("employees.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print("\n=== CSV File Created ===")

# Read CSV file
with open("employees.csv", "r") as file:
    reader = csv.reader(file)
    headers = next(reader)  # Get headers
    
    print(f"Headers: {headers}")
    print("\nEmployee Data:")
    for row in reader:
        print(f"  {row[0]} ({row[1]} years old) - {row[3]} in {row[2]}")

# Read CSV as dictionary
with open("employees.csv", "r") as file:
    dict_reader = csv.DictReader(file)
    
    print("\n=== CSV as Dictionary ===")
    for row in dict_reader:
        print(f"{row['Name']}: {row['Occupation']} in {row['City']}")

# =============================================================================
# 7. PRACTICAL EXERCISES
# =============================================================================

print("\n" + "="*50)
print("PRACTICE TIME!")
print("="*50)

# Exercise 1: Create a simple note-taking system
class NoteManager:
    def __init__(self, filename="notes.json"):
        self.filename = filename
        self.notes = self.load_notes()
    
    def load_notes(self):
        try:
            with open(self.filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            print("Warning: Notes file corrupted, starting fresh")
            return []
    
    def save_notes(self):
        try:
            with open(self.filename, "w") as file:
                json.dump(self.notes, file, indent=2)
            return True
        except Exception as e:
            print(f"Error saving notes: {e}")
            return False
    
    def add_note(self, title, content):
        note = {
            "id": len(self.notes) + 1,
            "title": title,
            "content": content,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat()
        }
        self.notes.append(note)
        if self.save_notes():
            return f"Note '{title}' added successfully!"
        return "Failed to save note"
    
    def list_notes(self):
        if not self.notes:
            return "No notes found"
        
        result = "Your Notes:\n"
        for note in self.notes:
            result += f"  {note['id']}. {note['title']}\n"
        return result
    
    def get_note(self, note_id):
        for note in self.notes:
            if note['id'] == note_id:
                return f"Title: {note['title']}\nContent: {note['content']}\nCreated: {note['created']}"
        return "Note not found"

# Test the note manager
notes = NoteManager()
print(notes.add_note("Python Learning", "Started learning Python today. It's amazing!"))
print(notes.add_note("Project Ideas", "1. Build a calculator\n2. Create a todo app\n3. Make a web scraper"))
print(notes.list_notes())
print("\n" + notes.get_note(1))

# Exercise 2: File statistics analyzer
def analyze_text_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
            
        # Calculate statistics
        lines = content.split('\n')
        words = content.split()
        characters = len(content)
        characters_no_spaces = len(content.replace(' ', ''))
        
        # Word frequency
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        # Most common word
        most_common = max(word_freq.items(), key=lambda x: x[1]) if word_freq else ("", 0)
        
        stats = {
            "filename": filename,
            "lines": len(lines),
            "words": len(words),
            "characters": characters,
            "characters_no_spaces": characters_no_spaces,
            "most_common_word": most_common[0],
            "most_common_count": most_common[1],
            "unique_words": len(word_freq)
        }
        
        return stats
        
    except FileNotFoundError:
        return {"error": f"File '{filename}' not found"}
    except Exception as e:
        return {"error": f"Error analyzing file: {e}"}

# Analyze our sample file
stats = analyze_text_file("sample.txt")
if "error" not in stats:
    print(f"\n=== File Analysis: {stats['filename']} ===")
    print(f"Lines: {stats['lines']}")
    print(f"Words: {stats['words']}")
    print(f"Characters: {stats['characters']}")
    print(f"Unique words: {stats['unique_words']}")
    print(f"Most common word: '{stats['most_common_word']}' ({stats['most_common_count']} times)")
else:
    print(f"Error: {stats['error']}")

# Clean up created files
cleanup_files = ["sample.txt", "tips.txt", "students.json", "nonexistent.txt", 
                "app.log", "config.json", "employees.csv", "notes.json"]

print(f"\n=== Cleaning up {len(cleanup_files)} temporary files ===")
for filename in cleanup_files:
    try:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"✅ Removed {filename}")
    except Exception as e:
        print(f"❌ Could not remove {filename}: {e}")

print("\n🎉 Excellent! You've mastered file handling and error management!")
print("Next: Run this file and then move to 06_useful_libraries.py")
