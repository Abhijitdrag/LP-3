
# DAA Practical: Huffman Encoding Using Greedy Strategy
# Question : Write a program to implement Huffman Encoding using a greedy strategy.
# Huffman Encoding is a lossless data compression technique.
# It assigns shorter binary codes to frequently occurring characters
# and longer binary codes to less frequent characters.

# Greedy Strategy:
# 1. Select two nodes having the lowest frequencies.
# 2. Combine them into a new node.
# 3. Add the new node back into the priority queue.
# 4. Repeat until only one node remains.

import heapq

# Function to generate Huffman codes
def generate_codes(node, code, codes):

    # If node is a character, store its binary code
    if isinstance(node, str):
        codes[node] = code if code else "0"
        return

    # Left child gets 0 and right child gets 1
    left, right = node

    generate_codes(left, code + "0", codes)
    generate_codes(right, code + "1", codes)


# Main program
n = int(input("Enter number of characters: "))

if n <= 0:
    print("Number of characters must be positive.")

else:
    heap = []
    visited = set()
    valid = True

    # Accept characters and their frequencies
    for i in range(n):
        ch = input("Enter character: ")
        freq = int(input("Enter frequency: "))

        if len(ch) == 0 or ch in visited or freq <= 0:
            print("Invalid or duplicate character/frequency.")
            valid = False
            break

        visited.add(ch)

        # Unique counter avoids comparing tree nodes
        heapq.heappush(heap, (freq, i, ch))

    if valid:

        count = n

        # Combine the two minimum-frequency nodes
        while len(heap) > 1:

            f1, _, left = heapq.heappop(heap)
            f2, _, right = heapq.heappop(heap)

            # Create a new internal node
            new_node = (left, right)

            heapq.heappush(heap, (f1 + f2, count, new_node))
            count += 1

        # Remaining node is the Huffman tree root
        root = heap[0][2]

        codes = {}

        # Generate binary codes
        generate_codes(root, "", codes)

        print("\nHuffman Codes:")
        for ch, code in codes.items():
            print(ch, ":", code)


# ==============================================
# IMPORTANT THEORY AND VIVA QUESTIONS
# ==============================================

# 1. What is Huffman Encoding?
# It is a lossless data compression algorithm.
# It reduces the number of bits required to represent data.

# 2. Which algorithmic technique is used?
# Greedy Algorithm.

# 3. Why is it called a greedy algorithm?
# Because it always selects the two nodes
# having the minimum frequencies at each step.

# 4. What is a Priority Queue?
# A data structure that removes the element
# with the highest priority.
# Here, the smallest frequency has the highest priority.
# Python heapq implements a min-heap.

# 5. What is a Huffman Tree?
# It is a binary tree used to generate Huffman codes.
# Leaf nodes represent characters.
# Internal nodes represent combined frequencies.

# 6. What do 0 and 1 represent?
# Left edge  = 0
# Right edge = 1

# 7. What is a Prefix-Free Code?
# No character's code is the prefix of another
# character's code.
# This makes decoding unambiguous.

# 8. Time Complexity:
# O(n log n)
# n = number of distinct characters.
# Each heap insertion/removal takes O(log n).
# There are O(n) heap operations.

# 9. Space Complexity:
# O(n)
# Huffman tree and code dictionary require O(n) space.
# Recursive tree traversal can also use O(n) stack space.

# 10. Is Huffman Encoding lossless?
# Yes. Original data can be recovered without data loss.

# 11. What happens if two frequencies are equal?
# Either node may be selected first.
# Different valid Huffman codes may be generated.

# 12. What if there is only one character?
# Its Huffman code is assigned as 0.

# 13. What is the main advantage?
# It reduces storage space by assigning variable-length codes.

# 14. Difference between fixed-length and variable-length coding?
# Fixed-length: Every character uses the same number of bits.
# Variable-length: Characters can use different numbers of bits.

# 15. Why is Huffman Encoding used?
# File compression, ZIP-like compression techniques,
# image compression and data transmission.

# IMPORTANT:
# Huffman codes are not necessarily unique when frequencies
# are equal, but all valid generated codes remain prefix-free.

# ==============================================
# END OF PRACTICAL
# ==============================================
