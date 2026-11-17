
# DAA Practical: Job Sequencing with Deadlines
# Using Greedy Strategy
# Question : Implement job sequencing with deadlines using a greedy method.
# Objective:
# Schedule jobs to maximize total profit.
#
# Every job has:
# 1. Job ID
# 2. Deadline
# 3. Profit
#
# Assumptions:
# - Every job requires exactly one unit of time.
# - Only one job can be performed in one time slot.
# - A job earns profit only if completed by its deadline.
#
# Greedy Strategy:
# 1. Sort jobs in decreasing order of profit.
# 2. Select the job with the highest profit.
# 3. Place it in the latest available slot
#    on or before its deadline.
# 4. Repeat for remaining jobs.

# Main program
n = int(input("Enter number of jobs: "))

jobs = []

# Accept job details
for i in range(n):

    job_id = input("\nEnter Job ID: ")
    deadline = int(input("Enter Deadline: "))
    profit = int(input("Enter Profit: "))

    jobs.append((job_id, deadline, profit))


if n <= 0:
    print("No jobs available.")

else:

    # Sort jobs according to profit in descending order
    jobs.sort(key=lambda x: x[2], reverse=True)

    # Maximum useful deadline cannot exceed number of jobs
    max_deadline = min(
        max((d for _, d, p in jobs if d > 0 and p > 0), default=0),
        n
    )

    # Create empty time slots
    slots = [None] * (max_deadline + 1)

    total_profit = 0

    # Process jobs in decreasing order of profit
    for job_id, deadline, profit in jobs:

        # Ignore jobs with invalid deadlines or non-positive profit
        if deadline <= 0 or profit <= 0:
            continue

        # Find the latest available slot before the deadline
        for j in range(min(deadline, max_deadline), 0, -1):

            if slots[j] is None:

                # Assign job to this slot
                slots[j] = job_id

                # Add job profit
                total_profit += profit

                break

    # Display scheduled jobs
    print("\nScheduled Jobs:")

    for i in range(1, max_deadline + 1):

        if slots[i] is not None:
            print("Slot", i, ":", slots[i])

    print("\nMaximum Total Profit:", total_profit)


# ==============================================
# IMPORTANT THEORY AND VIVA QUESTIONS
# ==============================================

# 1. What is Job Sequencing with Deadlines?
# It is an optimization problem in which jobs
# are scheduled to maximize total profit
# while satisfying their deadlines.

# 2. Which algorithmic technique is used?
# Greedy Algorithm.

# 3. What is the greedy strategy?
# Select the job with the highest profit first
# and assign it to the latest available slot
# before or on its deadline.

# 4. Why do we sort jobs by profit?
# To consider the most profitable jobs first.

# 5. Why do we schedule a job in the latest slot?
# It keeps earlier slots available for jobs
# having smaller deadlines.

# 6. What is a deadline?
# The latest time by which a job must be completed.

# 7. What is a time slot?
# A time interval in which one job can be completed.
# Every job requires one unit of time.

# 8. Time Complexity:
# Sorting: O(n log n)
# Scheduling: O(n * D)
# Overall: O(n log n + n * D)
#
# D = maximum useful deadline.
# Since D is capped at n:
# Worst-case time complexity = O(n^2).

# 9. Space Complexity:
# O(n)
# Jobs list and time slots require O(n) space.

# 10. What is the objective?
# Maximize total profit while satisfying deadlines.

# 11. What happens if a slot is already occupied?
# The algorithm checks the previous available slot.

# 12. Can multiple jobs execute in one slot?
# No. Only one job can be scheduled per slot.

# 13. What happens if no slot is available?
# That job is skipped.

# 14. What happens when two jobs have equal profit?
# Either job can be considered first.
# The final schedule may differ, but it remains valid.

# 15. Is this algorithm optimal?
# For the standard job sequencing problem with
# unit-time jobs and deadlines, this greedy strategy
# produces an optimal maximum-profit schedule.

# 16. What is the role of sorting?
# Sorting ensures jobs are considered
# from highest profit to lowest profit.

# 17. What is an optimization problem?
# A problem that aims to find the best solution
# among all possible valid solutions.

# 18. What is the main advantage of the greedy method?
# It provides an optimal solution for this
# standard problem with a simple approach.

# IMPORTANT:
# The algorithm assumes independent jobs,
# unit execution time, and no preemption.

# ==============================================
# END OF PRACTICAL
# ==============================================
