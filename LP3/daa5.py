
# DAA Practical: 8 Queens Problem Using Backtracking
# Question : Design 8-Queens matrix having first Queen placed. Use backtracking to place remaining Queens to generate the final 8-queen’s matrix.
# Objective:
# Place 8 queens on an 8x8 chessboard such that
# no two queens attack each other.
#
# First Queen is placed at a position entered by the user.
# Remaining 7 queens are placed using Backtracking.
#
# Rules:
# 1. No two queens can be in the same row.
# 2. No two queens can be in the same column.
# 3. No two queens can be on the same diagonal.
#
# Matrix:
# 1 = Queen is placed
# 0 = Empty position

N = 8

# Create an empty 8x8 chessboard
board = [[0 for j in range(N)] for i in range(N)]

# Take the position of the first queen from the user
row = int(input("Enter first queen row (1-8): ")) - 1
col = int(input("Enter first queen column (1-8): ")) - 1

# Check whether the given position is valid
if row < 0 or row >= N or col < 0 or col >= N:

    print("Invalid position! Enter row and column between 1 and 8.")

else:

    # Place the first queen
    board[row][col] = 1

    # Sets store already occupied columns and diagonals.
    # This helps check safety in O(1) average time.
    columns = {col}
    diagonal1 = {row - col}
    diagonal2 = {row + col}

    # Recursive function to place remaining queens
    def solve(r):

        # All 8 rows have been processed
        if r == N:
            return True

        # Skip the row where the first queen is already placed
        if r == row:
            return solve(r + 1)

        # Try placing a queen in every column
        for c in range(N):

            # Check whether the position is safe
            if (c not in columns and
                (r - c) not in diagonal1 and
                (r + c) not in diagonal2):

                # Place the queen
                board[r][c] = 1

                # Mark column and diagonals as occupied
                columns.add(c)
                diagonal1.add(r - c)
                diagonal2.add(r + c)

                # Recursively place the next queen
                if solve(r + 1):
                    return True

                # BACKTRACKING:
                # If no solution is found, remove the queen
                board[r][c] = 0

                columns.remove(c)
                diagonal1.remove(r - c)
                diagonal2.remove(r + c)

        # No safe position found in this row
        return False

    # Call the backtracking function
    if solve(0):

        print("\n8 Queens Solution:")
        print("\nFinal 8x8 Chessboard Matrix:")

        # Display the final matrix
        for i in range(N):
            for j in range(N):
                print(board[i][j], end=" ")
            print()

        print("\nAll 8 queens are placed successfully!")

    else:
        print("\nNo solution exists for the given first queen position.")


# =====================================================
# IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
# =====================================================

# 1. What is the 8 Queens Problem?
# It is a chessboard problem in which 8 queens must
# be placed on an 8x8 board without attacking each other.

# 2. Which algorithmic technique is used?
# Backtracking.

# 3. What is Backtracking?
# Backtracking is an algorithmic technique that builds
# a solution step by step and removes a choice when
# that choice cannot lead to a valid solution.

# 4. Why is Backtracking used?
# Because it checks possible queen positions and
# rejects unsafe positions without exploring further.

# 5. What are the conditions for placing a queen?
# - No other queen in the same row.
# - No other queen in the same column.
# - No other queen in the same diagonal.

# 6. How do we identify diagonal positions?
# Two queens are on the same diagonal if:
#
# Row difference = Column difference
# OR
# (row - column) values are equal.
#
# For the other diagonal:
# (row + column) values are equal.

# 7. What is the role of the solve() function?
# It recursively tries to place queens
# row by row until all 8 queens are placed.

# 8. What is the base condition?
# If row == 8, all 8 queens have been placed.
# Return True to indicate a successful solution.

# 9. What is the role of the for loop?
# It checks all 8 columns in the current row
# to find a safe position for the queen.

# 10. What happens when a position is unsafe?
# That position is skipped and the next column is checked.

# 11. What happens when no valid position is available?
# The algorithm backtracks to the previous row,
# removes the previously placed queen,
# and tries another position.

# 12. What is the purpose of backtracking?
# To undo an incorrect choice and try another possibility.

# 13. What is the role of the columns set?
# It stores columns that already contain a queen
# to avoid placing another queen in the same column.

# 14. What are diagonal1 and diagonal2?
# diagonal1 stores row - column.
# diagonal2 stores row + column.
# These sets help detect diagonal conflicts.

# 15. Why do we use sets?
# Sets provide O(1) average-time membership checking
# and make safety checking faster.

# 16. Time Complexity:
# O(N!) is a commonly used upper bound for
# the row-wise backtracking approach.
# Here N = 8.
# For a general N-Queens problem, the upper bound
# is O(N!) with constant-time conflict checks.

# 17. Space Complexity:
# O(N^2) including the chessboard matrix.
# O(N) auxiliary space for recursion and sets.
#
# For N = 8, the board contains 64 cells.

# 18. What is a state-space tree?
# It represents all possible choices made by
# the backtracking algorithm.
# Each level represents a row and each branch
# represents a possible queen position.

# 19. What is pruning?
# Pruning means rejecting an invalid choice
# without exploring its remaining possibilities.
# Here, unsafe queen positions are pruned.

# 20. What is recursion?
# Recursion is a technique in which a function
# calls itself to solve smaller subproblems.

# 21. What is the difference between Backtracking
# and Brute Force?
#
# Brute Force:
# Tries all possible combinations.
#
# Backtracking:
# Rejects invalid combinations early
# and avoids unnecessary exploration.

# 22. Can two queens be placed in the same row?
# No. This implementation places exactly one queen
# in each row.

# 23. Can the first queen be placed anywhere?
# Yes, the program accepts the first queen's position.
# However, a complete solution may not exist
# for every possible fixed position in general.

# 24. How many queens are placed by the algorithm?
# One queen is already placed by the user.
# The algorithm places the remaining seven queens.

# 25. Is the 8 Queens Problem an optimization problem?
# No. It is primarily a constraint satisfaction problem.
# The objective is to find a valid arrangement.

# 26. Is the solution unique?
# No. The 8 Queens Problem has 92 distinct solutions
# when rotations and reflections are counted separately.
# It has 12 fundamental solutions when symmetric
# arrangements are treated as equivalent.

# 27. What is the role of the board matrix?
# It stores the final queen positions.
# 1 represents a queen and 0 represents an empty cell.

# 28. Important algorithm steps:
# Step 1: Create an empty 8x8 matrix.
# Step 2: Accept the first queen's position.
# Step 3: Place the first queen.
# Step 4: Check safe positions in the next row.
# Step 5: Place the queen in a safe position.
# Step 6: Recursively process the next row.
# Step 7: Backtrack if no safe position is found.
# Step 8: Print the final matrix when all queens are placed.

# =====================================================
# SAMPLE INPUT:
#
# Enter first queen row (1-8): 1
# Enter first queen column (1-8): 1
#
# SAMPLE OUTPUT:
#
# 1 0 0 0 0 0 0 0
# 0 0 0 0 1 0 0 0
# 0 0 0 0 0 0 0 1
# 0 0 0 0 0 1 0 0
# 0 0 1 0 0 0 0 0
# 0 0 0 0 0 0 1 0
# 0 1 0 0 0 0 0 0
# 0 0 0 1 0 0 0 0
#
# All 8 queens are placed successfully!
#
# =====================================================
# COMPLEXITY SUMMARY:
#
# Algorithm             : Backtracking
# Problem Type          : Constraint Satisfaction
# Board Size            : 8 x 8
# Time Complexity       : O(N!) upper bound
# Space Complexity      : O(N^2) including board
# Auxiliary Space       : O(N)
# Data Structures       : 2D List and Sets
#
# IMPORTANT EXTERNAL VIVA POINT:
# Backtracking places queens row by row, checks
# column and diagonal constraints, and removes a queen
# whenever the current arrangement cannot be completed.
#
# =====================================================
# END OF PRACTICAL
# =====================================================
