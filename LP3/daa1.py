
# DAA Practical: Fibonacci Numbers and Step Count
# Question : Write a program to calculate Fibonacci numbers and find its step count.
# Fibonacci Series:
# 0, 1, 1, 2, 3, 5, 8, 13, ...
# Each number is the sum of the previous two numbers.

# Function to calculate Fibonacci numbers
def fibonacci(n):
    a = 0
    b = 1
    steps = 0

    print("Fibonacci Series:")

    for i in range(n):
        print(a, end=" ")

        # Calculate the next Fibonacci number
        c = a + b
        a = b
        b = c

        # Count basic operations
        steps += 1

    print("\nTotal Step Count:", steps)


# Main program
n = int(input("Enter the number of terms: "))

if n < 0:
    print("Please enter a non-negative number.")
else:
    fibonacci(n)


# ==========================================
# IMPORTANT THEORY / VIVA QUESTIONS
# ==========================================

# 1. What is Fibonacci Series?
# A series where each number is the sum of
# the previous two numbers.
# Starting values are 0 and 1.

# 2. What is the algorithm used?
# Iterative approach using a for loop.

# 3. What is Step Count?
# Step count represents the number of times
# a particular operation or instruction executes.
# Here, steps += 1 executes n times.
# Therefore, step count = n.

# 4. Time Complexity:
# O(n) - The loop executes n times.

# 5. Space Complexity:
# O(1) - Only three variables (a, b, c)
# and a counter are used, regardless of n.
# Output storage is not counted.

# 6. Best Case:
# O(n) for n terms, since all terms are generated.

# 7. Worst Case:
# O(n) for n terms.

# 8. Auxiliary Space:
# O(1) - Constant extra memory.

# 9. Which approach is better: Iterative or Recursive?
# Iterative is better for this problem because:
# - Uses less memory.
# - Avoids repeated calculations.
# - Does not cause recursion stack overhead.

# 10. Recursive Fibonacci Complexity:
# Time Complexity: O(2^n) approximately.
# Space Complexity: O(n) due to recursion stack.
# Recursive Fibonacci repeats calculations.

# 11. What is the base case?
# The first two Fibonacci numbers are 0 and 1.

# 12. Why is Fibonacci used in DAA?
# To understand recursion, iteration,
# algorithm analysis, time complexity and
# space complexity.

# 13. What happens when n = 0?
# No terms are printed and step count is 0.

# 14. What happens when n = 1?
# Only 0 is printed and step count is 1.

# 15. Is this algorithm efficient?
# Yes. It generates n terms in linear time O(n)
# with constant auxiliary space O(1).

# ==========================================
# END OF PRACTICAL
# ==========================================
