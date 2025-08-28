# Mini Projects - Apply Your Python Skills!
# These projects will help you practice everything you've learned

import random
import json
import datetime
from collections import Counter
import os

print("🚀 Welcome to Python Mini Projects!")
print("These projects will help you practice and apply your skills.\n")

# =============================================================================
# PROJECT 1: NUMBER GUESSING GAME
# =============================================================================

def number_guessing_game():
    """A fun number guessing game with different difficulty levels."""
    
    print("="*50)
    print("🎯 PROJECT 1: NUMBER GUESSING GAME")
    print("="*50)
    
    # Choose difficulty
    difficulties = {
        "easy": (1, 10, 5),
        "medium": (1, 50, 7),
        "hard": (1, 100, 10)
    }
    
    print("Choose difficulty:")
    for level, (min_num, max_num, attempts) in difficulties.items():
        print(f"  {level.title()}: {min_num}-{max_num} ({attempts} attempts)")
    
    # For demo, we'll use medium difficulty
    min_num, max_num, max_attempts = difficulties["medium"]
    secret_number = random.randint(min_num, max_num)
    attempts = 0
    
    print(f"\n🎮 Game Started! Guess the number between {min_num} and {max_num}")
    print(f"You have {max_attempts} attempts.\n")
    
    # Simulate some guesses for demo
    demo_guesses = [25, 35, 42, 38]  # Let's say secret number is 40
    secret_number = 40  # Set for demo
    
    for guess in demo_guesses:
        attempts += 1
        print(f"Attempt {attempts}: Guess = {guess}")
        
        if guess == secret_number:
            print(f"🎉 Congratulations! You guessed it in {attempts} attempts!")
            break
        elif guess < secret_number:
            print("📈 Too low! Try higher.")
        else:
            print("📉 Too high! Try lower.")
        
        if attempts >= max_attempts:
            print(f"💔 Game over! The number was {secret_number}")
            break
        
        print(f"Attempts remaining: {max_attempts - attempts}\n")
    
    return attempts

# Run the game
game_attempts = number_guessing_game()

# =============================================================================
# PROJECT 2: TODO LIST MANAGER
# =============================================================================

class TodoManager:
    """A comprehensive todo list manager with file persistence."""
    
    def __init__(self, filename="todos.json"):
        self.filename = filename
        self.todos = self.load_todos()
        self.next_id = max([todo['id'] for todo in self.todos], default=0) + 1
    
    def load_todos(self):
        """Load todos from file."""
        try:
            with open(self.filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            print("Warning: Todo file corrupted, starting fresh")
            return []
    
    def save_todos(self):
        """Save todos to file."""
        try:
            with open(self.filename, 'w') as file:
                json.dump(self.todos, file, indent=2)
            return True
        except Exception as e:
            print(f"Error saving todos: {e}")
            return False
    
    def add_todo(self, task, priority="medium"):
        """Add a new todo item."""
        todo = {
            "id": self.next_id,
            "task": task,
            "priority": priority,
            "completed": False,
            "created": datetime.datetime.now().isoformat(),
            "completed_date": None
        }
        self.todos.append(todo)
        self.next_id += 1
        self.save_todos()
        return f"✅ Added: '{task}' (Priority: {priority})"
    
    def complete_todo(self, todo_id):
        """Mark a todo as completed."""
        for todo in self.todos:
            if todo['id'] == todo_id:
                todo['completed'] = True
                todo['completed_date'] = datetime.datetime.now().isoformat()
                self.save_todos()
                return f"🎉 Completed: '{todo['task']}'"
        return f"❌ Todo with ID {todo_id} not found"
    
    def list_todos(self, show_completed=False):
        """List all todos."""
        if not self.todos:
            return "📝 No todos found. Add some tasks!"
        
        result = "\n📋 YOUR TODO LIST:\n" + "="*30 + "\n"
        
        # Group by priority
        priority_order = {"high": 1, "medium": 2, "low": 3}
        sorted_todos = sorted(self.todos, key=lambda x: (
            x['completed'], 
            priority_order.get(x['priority'], 4),
            x['id']
        ))
        
        for todo in sorted_todos:
            if not show_completed and todo['completed']:
                continue
                
            status = "✅" if todo['completed'] else "⏳"
            priority_icon = {"high": "🔥", "medium": "📋", "low": "💤"}.get(todo['priority'], "📋")
            
            result += f"{status} [{todo['id']}] {priority_icon} {todo['task']}"
            if todo['completed'] and todo['completed_date']:
                completed_date = datetime.datetime.fromisoformat(todo['completed_date'])
                result += f" (Completed: {completed_date.strftime('%m/%d/%Y')})"
            result += "\n"
        
        return result
    
    def get_statistics(self):
        """Get todo statistics."""
        total = len(self.todos)
        completed = sum(1 for todo in self.todos if todo['completed'])
        pending = total - completed
        
        priority_count = Counter(todo['priority'] for todo in self.todos if not todo['completed'])
        
        stats = f"""
📊 TODO STATISTICS:
  Total tasks: {total}
  Completed: {completed}
  Pending: {pending}
  
  Pending by priority:
    🔥 High: {priority_count.get('high', 0)}
    📋 Medium: {priority_count.get('medium', 0)}
    💤 Low: {priority_count.get('low', 0)}
"""
        return stats

def todo_demo():
    """Demonstrate the todo manager."""
    print("="*50)
    print("📝 PROJECT 2: TODO LIST MANAGER")
    print("="*50)
    
    # Create todo manager
    todo_manager = TodoManager("demo_todos.json")
    
    # Add sample todos
    sample_todos = [
        ("Learn Python basics", "high"),
        ("Build a web scraper", "medium"),
        ("Read about machine learning", "low"),
        ("Practice coding daily", "high"),
        ("Create a portfolio website", "medium")
    ]
    
    print("Adding sample todos...")
    for task, priority in sample_todos:
        print(f"  {todo_manager.add_todo(task, priority)}")
    
    print(todo_manager.list_todos())
    
    # Complete some todos
    print("\nCompleting some todos...")
    print(f"  {todo_manager.complete_todo(1)}")
    print(f"  {todo_manager.complete_todo(4)}")
    
    print(todo_manager.list_todos())
    print(todo_manager.get_statistics())
    
    # Cleanup
    if os.path.exists("demo_todos.json"):
        os.remove("demo_todos.json")
        print("🧹 Cleaned up demo file")

todo_demo()

# =============================================================================
# PROJECT 3: EXPENSE TRACKER
# =============================================================================

class ExpenseTracker:
    """Track and analyze personal expenses."""
    
    def __init__(self):
        self.expenses = []
        self.categories = ["Food", "Transportation", "Entertainment", "Shopping", 
                          "Bills", "Healthcare", "Education", "Other"]
    
    def add_expense(self, amount, description, category="Other"):
        """Add a new expense."""
        if category not in self.categories:
            category = "Other"
        
        expense = {
            "id": len(self.expenses) + 1,
            "amount": float(amount),
            "description": description,
            "category": category,
            "date": datetime.date.today().isoformat()
        }
        
        self.expenses.append(expense)
        return f"💰 Added expense: ${amount} for {description}"
    
    def get_expenses_by_category(self):
        """Group expenses by category."""
        category_totals = {}
        for expense in self.expenses:
            category = expense['category']
            category_totals[category] = category_totals.get(category, 0) + expense['amount']
        
        return category_totals
    
    def get_monthly_summary(self):
        """Get summary for current month."""
        current_month = datetime.date.today().strftime("%Y-%m")
        monthly_expenses = [
            exp for exp in self.expenses 
            if exp['date'].startswith(current_month)
        ]
        
        total = sum(exp['amount'] for exp in monthly_expenses)
        count = len(monthly_expenses)
        avg = total / count if count > 0 else 0
        
        return {
            "total": total,
            "count": count,
            "average": avg,
            "expenses": monthly_expenses
        }
    
    def generate_report(self):
        """Generate a comprehensive expense report."""
        if not self.expenses:
            return "📊 No expenses recorded yet!"
        
        total_spent = sum(exp['amount'] for exp in self.expenses)
        category_totals = self.get_expenses_by_category()
        monthly = self.get_monthly_summary()
        
        # Find most expensive category
        top_category = max(category_totals.items(), key=lambda x: x[1])
        
        report = f"""
💰 EXPENSE REPORT
{"="*40}
Total Spent: ${total_spent:.2f}
Number of Expenses: {len(self.expenses)}
Average per Expense: ${total_spent/len(self.expenses):.2f}

📊 BY CATEGORY:
"""
        
        # Sort categories by amount (descending)
        for category, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total_spent) * 100
            report += f"  {category}: ${amount:.2f} ({percentage:.1f}%)\n"
        
        report += f"""
📅 THIS MONTH:
  Monthly Total: ${monthly['total']:.2f}
  Monthly Count: {monthly['count']} expenses
  Monthly Average: ${monthly['average']:.2f}

🎯 INSIGHTS:
  Top Spending Category: {top_category[0]} (${top_category[1]:.2f})
  Daily Average: ${total_spent/30:.2f}
"""
        
        return report

def expense_tracker_demo():
    """Demonstrate the expense tracker."""
    print("="*50)
    print("💰 PROJECT 3: EXPENSE TRACKER")
    print("="*50)
    
    tracker = ExpenseTracker()
    
    # Add sample expenses
    sample_expenses = [
        (25.50, "Lunch at restaurant", "Food"),
        (60.00, "Gas for car", "Transportation"),
        (15.00, "Movie ticket", "Entertainment"),
        (120.00, "Groceries", "Food"),
        (45.00, "Internet bill", "Bills"),
        (80.00, "New shoes", "Shopping"),
        (30.00, "Coffee and snacks", "Food"),
        (200.00, "Electric bill", "Bills"),
        (35.00, "Uber ride", "Transportation")
    ]
    
    print("Adding sample expenses...")
    for amount, desc, category in sample_expenses:
        print(f"  {tracker.add_expense(amount, desc, category)}")
    
    print(tracker.generate_report())

expense_tracker_demo()

# =============================================================================
# PROJECT 4: PASSWORD MANAGER
# =============================================================================

class SimplePasswordManager:
    """A simple password manager with basic encryption (for demo purposes)."""
    
    def __init__(self):
        self.passwords = {}
        self.master_key = "demo_key_123"  # In real app, this would be user-provided
    
    def simple_encrypt(self, text):
        """Simple character shifting encryption (NOT secure for real use!)."""
        encrypted = ""
        for char in text:
            if char.isalnum():
                # Shift character by 3 positions
                if char.isdigit():
                    encrypted += str((int(char) + 3) % 10)
                elif char.islower():
                    encrypted += chr((ord(char) - ord('a') + 3) % 26 + ord('a'))
                elif char.isupper():
                    encrypted += chr((ord(char) - ord('A') + 3) % 26 + ord('A'))
            else:
                encrypted += char
        return encrypted
    
    def simple_decrypt(self, encrypted_text):
        """Simple decryption (reverse of encrypt)."""
        decrypted = ""
        for char in encrypted_text:
            if char.isalnum():
                if char.isdigit():
                    decrypted += str((int(char) - 3) % 10)
                elif char.islower():
                    decrypted += chr((ord(char) - ord('a') - 3) % 26 + ord('a'))
                elif char.isupper():
                    decrypted += chr((ord(char) - ord('A') - 3) % 26 + ord('A'))
            else:
                decrypted += char
        return decrypted
    
    def generate_strong_password(self, length=12):
        """Generate a strong password."""
        import string
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(random.choice(chars) for _ in range(length))
    
    def add_password(self, service, username, password=None):
        """Add a password for a service."""
        if not password:
            password = self.generate_strong_password()
        
        encrypted_password = self.simple_encrypt(password)
        
        self.passwords[service] = {
            "username": username,
            "password": encrypted_password,
            "created": datetime.datetime.now().isoformat()
        }
        
        return f"🔒 Password saved for {service} (username: {username})"
    
    def get_password(self, service):
        """Retrieve a password for a service."""
        if service in self.passwords:
            data = self.passwords[service]
            decrypted_password = self.simple_decrypt(data["password"])
            return {
                "service": service,
                "username": data["username"],
                "password": decrypted_password,
                "created": data["created"]
            }
        return None
    
    def list_services(self):
        """List all stored services."""
        if not self.passwords:
            return "🔐 No passwords stored yet!"
        
        result = "🔐 STORED PASSWORDS:\n" + "="*30 + "\n"
        for service, data in self.passwords.items():
            created_date = datetime.datetime.fromisoformat(data["created"])
            result += f"🌐 {service}\n"
            result += f"   Username: {data['username']}\n"
            result += f"   Created: {created_date.strftime('%Y-%m-%d %H:%M')}\n\n"
        
        return result
    
    def check_password_strength(self, password):
        """Check password strength."""
        score = 0
        feedback = []
        
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Use at least 8 characters")
        
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Include lowercase letters")
        
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Include uppercase letters")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Include numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Include special characters")
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
        strength = strength_levels[min(score, 4)]
        
        return {
            "score": score,
            "strength": strength,
            "feedback": feedback
        }

def password_manager_demo():
    """Demonstrate the password manager."""
    print("="*50)
    print("🔐 PROJECT 4: PASSWORD MANAGER")
    print("="*50)
    
    pm = SimplePasswordManager()
    
    # Add some passwords
    print("Adding passwords for various services...")
    services = [
        ("Gmail", "user@gmail.com"),
        ("Facebook", "john_doe"),
        ("Netflix", "movie_lover"),
        ("GitHub", "developer123")
    ]
    
    for service, username in services:
        generated_password = pm.generate_strong_password()
        print(f"  {pm.add_password(service, username, generated_password)}")
    
    print(f"\n{pm.list_services()}")
    
    # Demonstrate password retrieval
    print("Retrieving password for Gmail:")
    gmail_data = pm.get_password("Gmail")
    if gmail_data:
        print(f"  Service: {gmail_data['service']}")
        print(f"  Username: {gmail_data['username']}")
        print(f"  Password: {gmail_data['password']}")
    
    # Test password strength checker
    test_passwords = ["weak", "StrongPass123!", "12345678", "MySecureP@ssw0rd!"]
    print("\n🔍 PASSWORD STRENGTH ANALYSIS:")
    for pwd in test_passwords:
        result = pm.check_password_strength(pwd)
        print(f"  '{pwd}': {result['strength']} (Score: {result['score']}/5)")
        if result['feedback']:
            print(f"    Suggestions: {', '.join(result['feedback'])}")

password_manager_demo()

# =============================================================================
# PROJECT 5: SIMPLE CALCULATOR WITH HISTORY
# =============================================================================

class AdvancedCalculator:
    """An advanced calculator with history and multiple operations."""
    
    def __init__(self):
        self.history = []
        self.memory = 0
    
    def add(self, a, b):
        result = a + b
        self._save_to_history(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self._save_to_history(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self._save_to_history(f"{a} × {b} = {result}")
        return result
    
    def divide(self, a, b):
        if b == 0:
            error = "Error: Division by zero!"
            self._save_to_history(f"{a} ÷ {b} = {error}")
            return error
        result = a / b
        self._save_to_history(f"{a} ÷ {b} = {result}")
        return result
    
    def power(self, base, exponent):
        result = base ** exponent
        self._save_to_history(f"{base} ^ {exponent} = {result}")
        return result
    
    def square_root(self, number):
        if number < 0:
            error = "Error: Cannot calculate square root of negative number!"
            self._save_to_history(f"√{number} = {error}")
            return error
        import math
        result = math.sqrt(number)
        self._save_to_history(f"√{number} = {result}")
        return result
    
    def percentage(self, value, percent):
        result = (value * percent) / 100
        self._save_to_history(f"{percent}% of {value} = {result}")
        return result
    
    def _save_to_history(self, operation):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.history.append(f"[{timestamp}] {operation}")
    
    def get_history(self, last_n=None):
        if not self.history:
            return "📝 No calculations in history"
        
        history_to_show = self.history[-last_n:] if last_n else self.history
        
        result = "📊 CALCULATION HISTORY:\n" + "="*30 + "\n"
        for entry in history_to_show:
            result += f"  {entry}\n"
        
        return result
    
    def clear_history(self):
        self.history.clear()
        return "🗑️ History cleared!"
    
    def memory_store(self, value):
        self.memory = value
        return f"💾 Stored {value} in memory"
    
    def memory_recall(self):
        return self.memory
    
    def memory_clear(self):
        self.memory = 0
        return "🗑️ Memory cleared!"

def calculator_demo():
    """Demonstrate the advanced calculator."""
    print("="*50)
    print("🧮 PROJECT 5: ADVANCED CALCULATOR")
    print("="*50)
    
    calc = AdvancedCalculator()
    
    # Perform various calculations
    print("Performing sample calculations...")
    
    calculations = [
        ("Addition", lambda: calc.add(15, 25)),
        ("Subtraction", lambda: calc.subtract(100, 37)),
        ("Multiplication", lambda: calc.multiply(8, 7)),
        ("Division", lambda: calc.divide(144, 12)),
        ("Power", lambda: calc.power(2, 8)),
        ("Square Root", lambda: calc.square_root(64)),
        ("Percentage", lambda: calc.percentage(200, 15)),
        ("Division by Zero", lambda: calc.divide(10, 0))
    ]
    
    for operation, func in calculations:
        result = func()
        print(f"  {operation}: {result}")
    
    # Memory operations
    print(f"\n  {calc.memory_store(42)}")
    print(f"  Memory Recall: {calc.memory_recall()}")
    
    # Show history
    print(f"\n{calc.get_history()}")
    
    # Show last 3 calculations
    print(f"\n{calc.get_history(3)}")

calculator_demo()

# =============================================================================
# FINAL SUMMARY AND NEXT STEPS
# =============================================================================

def show_final_summary():
    """Show a summary of all completed projects."""
    print("\n" + "="*60)
    print("🎉 CONGRATULATIONS! YOU'VE COMPLETED ALL MINI PROJECTS!")
    print("="*60)
    
    projects_completed = [
        "🎯 Number Guessing Game - Random numbers, loops, conditionals",
        "📝 Todo List Manager - Classes, file I/O, JSON, datetime",
        "💰 Expense Tracker - Data analysis, categorization, reporting",
        "🔐 Password Manager - Security concepts, encryption basics",
        "🧮 Advanced Calculator - Mathematical operations, history tracking"
    ]
    
    skills_practiced = [
        "✅ Variables and data types",
        "✅ Control structures (if/else, loops)",
        "✅ Functions and classes",
        "✅ File handling and JSON",
        "✅ Error handling",
        "✅ Working with libraries",
        "✅ Object-oriented programming",
        "✅ Data structures and algorithms"
    ]
    
    print("\n🏆 PROJECTS COMPLETED:")
    for project in projects_completed:
        print(f"  {project}")
    
    print("\n💪 PYTHON SKILLS MASTERED:")
    for skill in skills_practiced:
        print(f"  {skill}")
    
    next_steps = [
        "🌐 Web Development (Flask/Django)",
        "📊 Data Science (pandas, numpy, matplotlib)",
        "🤖 Machine Learning (scikit-learn, TensorFlow)",
        "🔧 Automation (selenium, requests, APIs)",
        "🎮 Game Development (pygame)",
        "📱 GUI Applications (tkinter, PyQt)",
        "☁️ Cloud Computing (AWS, Google Cloud)",
        "🔗 API Development (FastAPI, REST APIs)"
    ]
    
    print(f"\n🚀 RECOMMENDED NEXT STEPS:")
    for step in next_steps:
        print(f"  {step}")
    
    print(f"\n💡 FINAL TIPS:")
    print(f"  • Practice coding every day, even if just 15 minutes")
    print(f"  • Build projects that solve real problems")
    print(f"  • Join Python communities (Reddit, Stack Overflow, Discord)")
    print(f"  • Read other people's code to learn new techniques")
    print(f"  • Don't be afraid to make mistakes - they're part of learning!")
    
    print(f"\n🎯 You now have a solid foundation in Python programming!")
    print(f"   Keep practicing, keep building, and most importantly, have fun! 🐍✨")

show_final_summary()
