
# DAA Practical: 0/1 Knapsack Using Dynamic Programming
# Question : Write a program to solve a 0-1 Knapsack problem using dynamic programming or branch and bound strategy.
# Objective:
# Maximize total profit without exceeding the knapsack capacity.
#
# In 0/1 Knapsack:
# - Each item can either be selected (1) or rejected (0).
# - Items cannot be divided into fractions.
# - Every item can be selected only once.
#
# Dynamic Programming:
# We solve smaller subproblems and store their results
# in a 2D table to avoid repeated calculations.

# Main Program
n = int(input("Enter number of items: "))

weights = []
profits = []

if n < 0:
    print("Number of items cannot be negative.")

else:

    # Accept weights and profits
    valid = True

    for i in range(n):

        print("\nItem", i + 1)

        w = int(input("Enter weight: "))
        p = int(input("Enter profit: "))

        if w <= 0 or p < 0:
            print("Invalid weight or profit.")
            valid = False
            break

        weights.append(w)
        profits.append(p)

    if valid:

        capacity = int(input("\nEnter knapsack capacity: "))

        if capacity < 0:
            print("Capacity cannot be negative.")

        else:

            # Create DP table with (n+1) rows
            # and (capacity+1) columns.
            # dp[i][w] = Maximum profit using first i items
            # with knapsack capacity w.

            dp = [[0 for j in range(capacity + 1)]
                  for i in range(n + 1)]

            # Fill the DP table
            for i in range(1, n + 1):

                for w in range(capacity + 1):

                    # Current item's weight and profit
                    wt = weights[i - 1]
                    profit = profits[i - 1]

                    # If item weight exceeds capacity,
                    # we cannot select the item.
                    if wt > w:
                        dp[i][w] = dp[i - 1][w]

                    else:
                        # Option 1: Do not select the item
                        not_take = dp[i - 1][w]

                        # Option 2: Select the item
                        take = profit + dp[i - 1][w - wt]

                        # Choose the option with maximum profit
                        dp[i][w] = max(take, not_take)

            # Maximum profit is stored in the last cell
            max_profit = dp[n][capacity]

            print("\nMaximum Profit:", max_profit)

            # Backtracking to find selected items
            selected = []
            w = capacity

            for i in range(n, 0, -1):

                # If value changed, current item was selected
                if dp[i][w] != dp[i - 1][w]:

                    selected.append(i)

                    # Reduce remaining capacity
                    w -= weights[i - 1]

            selected.reverse()

            print("\nSelected Items:")

            if len(selected) == 0:
                print("No items selected.")

            else:
                for i in selected:
                    print("Item", i,
                          "| Weight:", weights[i - 1],
                          "| Profit:", profits[i - 1])


# =====================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# =====================================================

# 1. What is the 0/1 Knapsack Problem?
# It is an optimization problem where we select items
# to maximize profit without exceeding capacity.
# Each item is either completely selected or rejected.

# 2. Which algorithmic technique is used?
# Dynamic Programming (DP).

# 3. Why is it called 0/1 Knapsack?
# 0 means the item is not selected.
# 1 means the item is selected.
# Fractional selection is not allowed.

# 4. What is Dynamic Programming?
# It is a technique that solves a complex problem
# by dividing it into smaller overlapping subproblems
# and storing their results.

# 5. What is the DP state?
# dp[i][w] represents the maximum profit obtained
# using the first i items with capacity w.

# 6. What is the recurrence relation?
#
# If weight[i-1] > w:
# dp[i][w] = dp[i-1][w]
#
# Otherwise:
# dp[i][w] = max(
#     dp[i-1][w],
#     profit[i-1] + dp[i-1][w-weight[i-1]]
# )
#
# First option: Do not select the item.
# Second option: Select the item.

# 7. What are the base conditions?
# dp[0][w] = 0
# dp[i][0] = 0
#
# No items or zero capacity means zero profit.

# 8. What is a DP Table?
# A 2D array used to store the results of subproblems.
# Rows represent items.
# Columns represent knapsack capacity.

# 9. Time Complexity:
# O(n * W)
# n = Number of items.
# W = Knapsack capacity.
# Two nested loops fill the DP table.

# 10. Space Complexity:
# O(n * W)
# A 2D DP table of size (n+1) * (W+1) is created.

# 11. Best Case:
# O(n * W)

# 12. Average Case:
# O(n * W)

# 13. Worst Case:
# O(n * W)
# Every DP table cell is calculated.

# 14. What is Overlapping Subproblems?
# The same smaller problems are solved repeatedly.
# Dynamic Programming stores their results to avoid
# unnecessary repeated calculations.

# 15. What is Optimal Substructure?
# An optimal solution can be constructed using
# optimal solutions of smaller subproblems.

# 16. Can we select a fraction of an item?
# No. Fractional selection is not allowed in 0/1 Knapsack.

# 17. Difference between 0/1 and Fractional Knapsack:
#
# 0/1 Knapsack:
# - Items cannot be divided.
# - Dynamic Programming is used for an optimal solution.
#
# Fractional Knapsack:
# - Items can be divided.
# - Greedy strategy using profit/weight ratio is optimal.

# 18. Why is Greedy not always optimal for 0/1 Knapsack?
# Selecting the highest profit/weight ratio first
# may prevent us from obtaining the maximum total profit.

# 19. What is Backtracking in this program?
# We move backwards through the DP table
# to identify which items were selected
# in the optimal solution.

# 20. What is the role of max()?
# It selects the maximum profit between
# including and excluding the current item.

# 21. Is this a polynomial-time algorithm?
# Its running time is O(n * W).
# It is called pseudo-polynomial because W is a
# numeric capacity value, not its number of bits.
# The running time can be exponential in the
# bit-length of the capacity.

# 22. Can Dynamic Programming solve large capacities efficiently?
# Large capacities may require a very large DP table.
# For such cases, other methods such as Branch and Bound
# may be considered.

# 23. What is Branch and Bound?
# It is an optimization technique that explores
# possible solutions using a state-space tree
# and prunes branches that cannot improve the
# current best solution.

# 24. Why do we use DP instead of simple recursion?
# DP stores previously calculated results,
# reducing repeated calculations.

# 25. What happens when capacity is zero?
# Maximum profit is zero and no items are selected.

# 26. Important assumptions:
# - Item weights are positive integers.
# - Profits are non-negative integers.
# - Each item can be selected only once.
# - Items cannot be divided.
# - Capacity is a non-negative integer.

# 27. Steps of the algorithm:
# Step 1: Read number of items.
# Step 2: Read weights and profits.
# Step 3: Read knapsack capacity.
# Step 4: Create a 2D DP table.
# Step 5: Fill the table using the recurrence relation.
# Step 6: Find maximum profit from dp[n][capacity].
# Step 7: Backtrack to identify selected items.
# Step 8: Display maximum profit and selected items.

# =====================================================
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
# Maximum Profit: 220
#
# Selected Items:
# Item 2 | Weight: 20 | Profit: 100
# Item 3 | Weight: 30 | Profit: 120
#
# =====================================================
# COMPLEXITY SUMMARY:
#
# Algorithm             : Dynamic Programming
# Data Structure        : 2D Array / List
# Time Complexity       : O(n * W)
# Best Case             : O(n * W)
# Average Case          : O(n * W)
# Worst Case            : O(n * W)
# Space Complexity      : O(n * W)
#
# IMPORTANT EXTERNAL VIVA POINT:
# 0/1 Knapsack uses Dynamic Programming because
# items cannot be divided. Fractional Knapsack uses
# a Greedy strategy because fractions are allowed.
#
# =====================================================
# END OF PRACTICAL
# =====================================================
