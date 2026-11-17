
# DAA Practical: Fractional Knapsack Using Greedy Method

# Objective:
# Maximize the total profit within the given knapsack capacity.
# Question : Write a program to solve a fractional Knapsack problem using a greedy method.
# Greedy Strategy:
# Select items having the highest profit/weight ratio first.
# If the complete item cannot fit, take only the required fraction.

# Function to solve Fractional Knapsack
def fractional_knapsack(items, capacity):

    # Sort items by profit/weight ratio in descending order
    items.sort(key=lambda x: x[1] / x[0], reverse=True)

    total_profit = 0

    print("\nSelected Items:")

    for weight, profit in items:

        # Stop if the knapsack is full
        if capacity <= 0:
            break

        # If the complete item fits
        if weight <= capacity:
            capacity -= weight
            total_profit += profit

            print("Weight:", weight, "| Fraction: 1 | Profit:", profit)

        # Otherwise, take only the required fraction
        else:
            fraction = capacity / weight
            earned_profit = profit * fraction

            total_profit += earned_profit

            print("Weight:", weight,
                  "| Fraction:", round(fraction, 2),
                  "| Profit:", round(earned_profit, 2))

            capacity = 0

    print("\nMaximum Profit:", round(total_profit, 2))


# Main Program
n = int(input("Enter number of items: "))
items = []

# Accept item weights and profits
for i in range(n):

    print("\nItem", i + 1)

    weight = float(input("Enter weight: "))
    profit = float(input("Enter profit: "))

    if weight <= 0 or profit < 0:
        print("Invalid weight or profit.")
        break

    items.append((weight, profit))

else:
    capacity = float(input("\nEnter knapsack capacity: "))

    if capacity < 0:
        print("Capacity cannot be negative.")
    else:
        fractional_knapsack(items, capacity)


# ==================================================
# IMPORTANT THEORY AND VIVA QUESTIONS
# ==================================================

# 1. What is the Knapsack Problem?
# It is an optimization problem in which items
# are selected to maximize profit within a given capacity.

# 2. What is Fractional Knapsack?
# In this problem, an item can be divided into fractions.
# We can take the complete item or a part of it.

# 3. Which algorithmic technique is used?
# Greedy Algorithm.

# 4. What is the greedy strategy?
# Select the item having the highest profit/weight ratio first.

# 5. What is the formula for profit/weight ratio?
# Ratio = Profit / Weight

# 6. Why do we sort in descending order?
# To select the most profitable item per unit weight first.

# 7. Can we take a fraction of an item?
# Yes. Fractional Knapsack allows items to be divided.

# 8. What is the main objective?
# Maximize total profit without exceeding knapsack capacity.

# 9. Time Complexity:
# Sorting items: O(n log n)
# Traversing items: O(n)
# Overall Time Complexity: O(n log n)

# 10. Space Complexity:
# O(n) for storing the input items.
# Sorting may also require O(n) auxiliary space
# in Python's implementation.

# 11. Best Case:
# O(n log n), because sorting is always required.

# 12. Worst Case:
# O(n log n), because sorting dominates the execution time.

# 13. What is the difference between Fractional and 0/1 Knapsack?
#
# Fractional Knapsack:
# - Items can be divided.
# - Greedy method gives an optimal solution.
#
# 0/1 Knapsack:
# - Items cannot be divided.
# - Either take the complete item or reject it.
# - Dynamic Programming is commonly used for an optimal solution.

# 14. Why is Greedy optimal for Fractional Knapsack?
# Because selecting the highest profit/weight ratio first
# always gives the maximum profit for the available capacity.

# 15. What happens when the complete item does not fit?
# The algorithm calculates the available capacity
# and takes only that fraction of the item.

# 16. What is an optimization problem?
# A problem that aims to find the best possible solution
# among multiple feasible solutions.

# 17. What is a feasible solution?
# A solution that satisfies all given constraints.
# Here, total selected weight must not exceed capacity.

# 18. What is the role of sorting?
# Sorting arranges items according to their
# profit/weight ratios from highest to lowest.

# 19. Can Greedy always solve every Knapsack problem optimally?
# No. Greedy gives an optimal solution for Fractional Knapsack,
# but not necessarily for 0/1 Knapsack.

# 20. What happens if capacity is zero?
# No items are selected and maximum profit is zero.

# 21. What is the greedy choice property?
# A globally optimal solution can be obtained by
# repeatedly making the locally best choice.
# For Fractional Knapsack, the best choice is the
# item with the highest profit/weight ratio.

# 22. Important assumptions:
# - Item weights must be positive.
# - Item profits must be non-negative.
# - Items can be divided into fractions.
# - Knapsack capacity cannot be exceeded.

# 23. Step-by-step working:
# Step 1: Read item weights and profits.
# Step 2: Calculate profit/weight ratio.
# Step 3: Sort items by ratio in descending order.
# Step 4: Select complete items while they fit.
# Step 5: Take a fraction of the next item if required.
# Step 6: Calculate and display maximum profit.

# ==================================================
# SAMPLE INPUT:
#
# Enter number of items: 3
#
# Item 1
# Enter weight: 10
# Enter profit: 60
#
# Item 2
# Enter weight: 20
# Enter profit: 100
#
# Item 3
# Enter weight: 30
# Enter profit: 120
#
# Enter knapsack capacity: 50
#
# SAMPLE OUTPUT:
#
# Selected Items:
# Weight: 10.0 | Fraction: 1 | Profit: 60.0
# Weight: 20.0 | Fraction: 1 | Profit: 100.0
# Weight: 30.0 | Fraction: 0.67 | Profit: 80.0
#
# Maximum Profit: 240.0
#
# ==================================================
# COMPLEXITY SUMMARY:
#
# Time Complexity       : O(n log n)
# Best Case             : O(n log n)
# Average Case          : O(n log n)
# Worst Case            : O(n log n)
# Space Complexity      : O(n)
# Algorithm             : Greedy
# Data Structure        : List
#
# IMPORTANT EXTERNAL VIVA POINT:
# Fractional Knapsack uses profit/weight ratio,
# whereas 0/1 Knapsack is commonly solved using
# Dynamic Programming for an optimal solution.
#
# ==================================================
# END OF PRACTICAL
# ==================================================
