// ============================================================
// BLOCKCHAIN TECHNOLOGY PRACTICAL
// Experiment: Student Data Management Using Solidity
//
// REQUIRED CONSTRUCTS:
// 1. Structures
// 2. Arrays
// 3. Fallback
//
// ADDITIONAL CONSTRUCT:
// 4. Receive Function
//
// IDE: Remix
// Compiler: Solidity 0.8.20
//
// TWO CONTRACTS ARE PROVIDED:
// 1. StudentData         - Simple practical version.
// 2. AdvancedStudentData - Extended external viva version.
//
// IMPORTANT:
// Both contracts have receive() and fallback().
// This resolves the compiler warning about a payable
// fallback function without a receive ether function.
// ============================================================


// ============================================================
// PART 1: SIMPLE STUDENT DATA CONTRACT
//
// Features:
// 1. Add Student.
// 2. Display Student.
// 3. Display Total Students.
// 4. Receive Ether.
// 5. Fallback Function.
//
// Constructs:
// Struct, Dynamic Array, Events, Receive and Fallback.
// ============================================================



// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract StudentData {

    // --------------------------------------------------------
    // STRUCTURE:
    // Groups different student information into one datatype.
    // --------------------------------------------------------

    struct Student {

        uint256 rollNo;

        string name;

        uint256 marks;
    }

    // --------------------------------------------------------
    // ARRAY:
    // Stores multiple student records.
    // --------------------------------------------------------

    Student[] public students;

    // Events record important contract operations.
    event StudentAdded(
        uint256 rollNo,
        string name,
        uint256 marks
    );

    event EtherReceived(
        address sender,
        uint256 amount
    );

    event FallbackCalled(
        address sender,
        uint256 amount,
        bytes data
    );

    // --------------------------------------------------------
    // FUNCTION 1: ADD STUDENT
    //
    // Accepts roll number, name and marks.
    // Push inserts a new student into the array.
    // --------------------------------------------------------

    function addStudent(
        uint256 _rollNo,
        string memory _name,
        uint256 _marks
    ) public {

        require(_rollNo > 0, "Invalid roll number");

        require(
            bytes(_name).length > 0,
            "Name required"
        );

        require(
            _marks <= 100,
            "Marks cannot exceed 100"
        );

        students.push(
            Student(_rollNo, _name, _marks)
        );

        emit StudentAdded(
            _rollNo,
            _name,
            _marks
        );
    }

    // --------------------------------------------------------
    // FUNCTION 2: DISPLAY STUDENT
    //
    // Uses array index to retrieve student information.
    // Index starts from 0.
    // --------------------------------------------------------

    function getStudent(uint256 index)
        public
        view
        returns (
            uint256,
            string memory,
            uint256
        )
    {

        require(
            index < students.length,
            "Invalid student index"
        );

        Student memory s = students[index];

        return (
            s.rollNo,
            s.name,
            s.marks
        );
    }

    // --------------------------------------------------------
    // FUNCTION 3: TOTAL STUDENTS
    //
    // Returns the total number of stored student records.
    // --------------------------------------------------------

    function getCount()
        public
        view
        returns (uint256)
    {

        return students.length;
    }

    // --------------------------------------------------------
    // FUNCTION 4: RECEIVE
    //
    // Executes when Ether is transferred with empty calldata.
    //
    // receive() is specifically used for plain Ether transfers.
    // --------------------------------------------------------

    receive() external payable {

        emit EtherReceived(
            msg.sender,
            msg.value
        );
    }

    // --------------------------------------------------------
    // FUNCTION 5: FALLBACK
    //
    // Executes when an unknown function selector is called.
    //
    // It also accepts Ether when calldata is non-empty.
    //
    // payable allows the contract to receive Ether.
    // --------------------------------------------------------

    fallback() external payable {

        emit FallbackCalled(
            msg.sender,
            msg.value,
            msg.data
        );
    }

}


// ============================================================
// PART 2: ADVANCED STUDENT DATA MANAGEMENT CONTRACT
//
// External Viva Extension:
//
// FEATURES:
//
// 1. Add Individual Student.
// 2. Add Multiple Students.
// 3. Display Student by Index.
// 4. Search Student by Roll Number.
// 5. Update Student Details.
// 6. Delete Student.
// 7. Display All Students.
// 8. Count Total Students.
// 9. Calculate Average Marks.
// 10. Find Topper.
// 11. Count Passing Students.
// 12. Count Failing Students.
// 13. Display Complete Class Summary.
// 14. Receive Ether.
// 15. Fallback Function.
// 16. Count Receive and Fallback Calls.
// 17. Display Contract Balance.
// 18. Owner-only Ether Withdrawal.
//
// ADDITIONAL CONSTRUCTS:
// Mapping, Modifier, Constructor, Loop and Events.
//
// ============================================================

contract AdvancedStudentData {

    // --------------------------------------------------------
    // STRUCTURE:
    // Stores complete student information.
    // --------------------------------------------------------

    struct Student {

        uint256 rollNo;

        string name;

        uint256 marks;
    }

    // --------------------------------------------------------
    // ARRAY:
    // Stores all student records.
    // --------------------------------------------------------

    Student[] private students;

    // --------------------------------------------------------
    // MAPPING:
    //
    // Stores student array position using roll number.
    //
    // Roll Number => Array Index + 1
    //
    // Zero means the student does not exist.
    // --------------------------------------------------------

    mapping(uint256 => uint256) private studentIndex;

    // Contract deployer becomes the owner.
    address public owner;

    // Track number of Ether receiving transactions.
    uint256 public receiveCallCount;

    // Track number of fallback transactions.
    uint256 public fallbackCallCount;

    // Reentrancy protection for Ether withdrawal.
    bool private withdrawalInProgress;

    // --------------------------------------------------------
    // EVENTS:
    // Events record important student and Ether operations.
    // --------------------------------------------------------

    event StudentAdded(
        uint256 rollNo,
        string name,
        uint256 marks
    );

    event StudentUpdated(
        uint256 rollNo,
        string name,
        uint256 marks
    );

    event StudentDeleted(
        uint256 rollNo
    );

    event EtherReceived(
        address indexed sender,
        uint256 amount
    );

    event FallbackExecuted(
        address indexed sender,
        uint256 amount,
        bytes data
    );

    event EtherWithdrawn(
        address indexed receiver,
        uint256 amount
    );

    // --------------------------------------------------------
    // CONSTRUCTOR:
    //
    // Executes once during deployment.
    // Sets the deployer's address as owner.
    // --------------------------------------------------------

    constructor() {

        owner = msg.sender;
    }

    // --------------------------------------------------------
    // MODIFIER:
    //
    // Restricts student management operations
    // to the contract owner.
    // --------------------------------------------------------

    modifier onlyOwner() {

        require(
            msg.sender == owner,
            "Only owner allowed"
        );

        _;
    }

    // --------------------------------------------------------
    // MODIFIER: REENTRANCY PROTECTION
    //
    // Prevents repeated entry into protected functions
    // during an external Ether transfer.
    // --------------------------------------------------------

    modifier nonReentrant() {

        require(
            !withdrawalInProgress,
            "Reentrant call"
        );

        withdrawalInProgress = true;

        _;

        withdrawalInProgress = false;
    }

    // --------------------------------------------------------
    // INTERNAL FUNCTION: ADD STUDENT
    //
    // Common insertion logic used by both
    // addStudent() and addMultipleStudents().
    // --------------------------------------------------------

    function _addStudent(
        uint256 _rollNo,
        string memory _name,
        uint256 _marks
    ) internal {

        require(
            _rollNo > 0,
            "Invalid roll number"
        );

        require(
            bytes(_name).length > 0,
            "Name required"
        );

        require(
            _marks <= 100,
            "Invalid marks"
        );

        // Prevent duplicate roll numbers.
        require(
            studentIndex[_rollNo] == 0,
            "Student already exists"
        );

        // Insert student into array.
        students.push(
            Student(
                _rollNo,
                _name,
                _marks
            )
        );

        // Store array position as index + 1.
        studentIndex[_rollNo] = students.length;

        emit StudentAdded(
            _rollNo,
            _name,
            _marks
        );
    }

    // --------------------------------------------------------
    // FUNCTION 1: ADD SINGLE STUDENT
    //
    // Adds one student record.
    // Only the owner can add student records.
    // --------------------------------------------------------

    function addStudent(
        uint256 _rollNo,
        string memory _name,
        uint256 _marks
    ) public onlyOwner {

        _addStudent(
            _rollNo,
            _name,
            _marks
        );
    }

    // --------------------------------------------------------
    // FUNCTION 2: ADD MULTIPLE STUDENTS
    //
    // Inserts multiple student records in one transaction.
    //
    // All three input arrays must have equal lengths.
    //
    // If any record is invalid, the entire transaction
    // is reverted, including previous insertions in that call.
    // --------------------------------------------------------

    function addMultipleStudents(
        uint256[] calldata rollNos,
        string[] calldata names,
        uint256[] calldata marks
    ) public onlyOwner {

        require(
            rollNos.length == names.length &&
            names.length == marks.length,
            "Array lengths must match"
        );

        require(
            rollNos.length > 0,
            "No students provided"
        );

        // Insert every student using a loop.
        for (
            uint256 i = 0;
            i < rollNos.length;
            i++
        ) {

            _addStudent(
                rollNos[i],
                names[i],
                marks[i]
            );
        }
    }

    // --------------------------------------------------------
    // FUNCTION 3: GET STUDENT BY INDEX
    //
    // Returns a complete record using array index.
    // --------------------------------------------------------

    function getStudentByIndex(
        uint256 index
    )
        public
        view
        returns (Student memory)
    {

        require(
            index < students.length,
            "Invalid index"
        );

        return students[index];
    }

    // --------------------------------------------------------
    // FUNCTION 4: SEARCH BY ROLL NUMBER
    //
    // Mapping provides direct access to student index.
    // --------------------------------------------------------

    function getStudentByRollNo(
        uint256 rollNo
    )
        public
        view
        returns (Student memory)
    {

        uint256 index = studentIndex[rollNo];

        require(
            index != 0,
            "Student not found"
        );

        return students[index - 1];
    }

    // --------------------------------------------------------
    // FUNCTION 5: UPDATE STUDENT
    //
    // Updates student name and marks using roll number.
    // --------------------------------------------------------

    function updateStudent(
        uint256 rollNo,
        string memory newName,
        uint256 newMarks
    ) public onlyOwner {

        uint256 index = studentIndex[rollNo];

        require(
            index != 0,
            "Student not found"
        );

        require(
            bytes(newName).length > 0,
            "Name required"
        );

        require(
            newMarks <= 100,
            "Invalid marks"
        );

        // Update existing student record.
        students[index - 1].name = newName;

        students[index - 1].marks = newMarks;

        emit StudentUpdated(
            rollNo,
            newName,
            newMarks
        );
    }

    // --------------------------------------------------------
    // FUNCTION 6: DELETE STUDENT
    //
    // Uses Swap-and-Pop to delete a student.
    //
    // The last element replaces the deleted element.
    // The last array element is then removed.
    //
    // NOTE:
    // Array order can change after deletion.
    // --------------------------------------------------------

    function deleteStudent(
        uint256 rollNo
    ) public onlyOwner {

        uint256 index = studentIndex[rollNo];

        require(
            index != 0,
            "Student not found"
        );

        uint256 arrayIndex = index - 1;

        uint256 lastIndex = students.length - 1;

        // Move last student into deleted position if needed.
        if (arrayIndex != lastIndex) {

            Student memory lastStudent = students[lastIndex];

            students[arrayIndex] = lastStudent;

            // Update moved student's mapping position.
            studentIndex[lastStudent.rollNo] = index;
        }

        // Remove deleted student's mapping.
        delete studentIndex[rollNo];

        // Remove last array element.
        students.pop();

        emit StudentDeleted(rollNo);
    }

    // --------------------------------------------------------
    // FUNCTION 7: DISPLAY ALL STUDENTS
    //
    // Returns all student records.
    // --------------------------------------------------------

    function getAllStudents()
        public
        view
        returns (Student[] memory)
    {

        return students;
    }

    // --------------------------------------------------------
    // FUNCTION 8: TOTAL STUDENTS
    // --------------------------------------------------------

    function getCount()
        public
        view
        returns (uint256)
    {

        return students.length;
    }

    // --------------------------------------------------------
    // FUNCTION 9: CALCULATE AVERAGE MARKS
    //
    // Adds all marks and divides by student count.
    // Uses integer division.
    // --------------------------------------------------------

    function getAverageMarks()
        public
        view
        returns (uint256)
    {

        require(
            students.length > 0,
            "No students available"
        );

        uint256 totalMarks = 0;

        for (
            uint256 i = 0;
            i < students.length;
            i++
        ) {

            totalMarks += students[i].marks;
        }

        return totalMarks / students.length;
    }

    // --------------------------------------------------------
    // FUNCTION 10: FIND TOPPER
    //
    // Returns the student having the highest marks.
    // --------------------------------------------------------

    function getTopper()
        public
        view
        returns (Student memory)
    {

        require(
            students.length > 0,
            "No students available"
        );

        uint256 maxIndex = 0;

        for (
            uint256 i = 1;
            i < students.length;
            i++
        ) {

            if (
                students[i].marks >
                students[maxIndex].marks
            ) {

                maxIndex = i;
            }
        }

        return students[maxIndex];
    }

    // --------------------------------------------------------
    // FUNCTION 11: COUNT PASSING STUDENTS
    //
    // Passing marks = 40 or above.
    // --------------------------------------------------------

    function getPassingCount()
        public
        view
        returns (uint256)
    {

        uint256 count = 0;

        for (
            uint256 i = 0;
            i < students.length;
            i++
        ) {

            if (students[i].marks >= 40) {

                count++;
            }
        }

        return count;
    }

    // --------------------------------------------------------
    // FUNCTION 12: COUNT FAILING STUDENTS
    //
    // Failing marks = below 40.
    // --------------------------------------------------------

    function getFailingCount()
        public
        view
        returns (uint256)
    {

        uint256 count = 0;

        for (
            uint256 i = 0;
            i < students.length;
            i++
        ) {

            if (students[i].marks < 40) {

                count++;
            }
        }

        return count;
    }

    // --------------------------------------------------------
    // FUNCTION 13: COMPLETE CLASS SUMMARY
    //
    // Returns:
    // 1. Total students.
    // 2. Total marks.
    // 3. Average marks.
    // 4. Passing students.
    // 5. Failing students.
    //
    // This is useful for external viva questions
    // about multiple student records.
    // --------------------------------------------------------

    function getClassSummary()
        public
        view
        returns (
            uint256 totalStudents,
            uint256 totalMarks,
            uint256 averageMarks,
            uint256 passingStudents,
            uint256 failingStudents
        )
    {

        totalStudents = students.length;

        require(
            totalStudents > 0,
            "No students available"
        );

        for (
            uint256 i = 0;
            i < students.length;
            i++
        ) {

            totalMarks += students[i].marks;

            if (students[i].marks >= 40) {

                passingStudents++;
            }
            else {

                failingStudents++;
            }
        }

        averageMarks = totalMarks / totalStudents;
    }

    // --------------------------------------------------------
    // FUNCTION 14: RECEIVE ETHER
    //
    // Executes when Ether is transferred with empty calldata.
    //
    // This is separate from fallback().
    // Adding receive() prevents the compiler warning
    // associated with a payable fallback without receive.
    // --------------------------------------------------------

    receive() external payable {

        receiveCallCount++;

        emit EtherReceived(
            msg.sender,
            msg.value
        );
    }

    // --------------------------------------------------------
    // FUNCTION 15: FALLBACK
    //
    // Executes for unknown function selectors.
    //
    // Can also receive Ether when calldata is non-empty.
    //
    // A plain Ether transfer with empty calldata
    // goes to receive(), not fallback().
    // --------------------------------------------------------

    fallback() external payable {

        fallbackCallCount++;

        emit FallbackExecuted(
            msg.sender,
            msg.value,
            msg.data
        );
    }

    // --------------------------------------------------------
    // FUNCTION 16: GET RECEIVE CALL COUNT
    // --------------------------------------------------------

    function getReceiveCount()
        public
        view
        returns (uint256)
    {

        return receiveCallCount;
    }

    // --------------------------------------------------------
    // FUNCTION 17: GET FALLBACK CALL COUNT
    // --------------------------------------------------------

    function getFallbackCount()
        public
        view
        returns (uint256)
    {

        return fallbackCallCount;
    }

    // --------------------------------------------------------
    // FUNCTION 18: DISPLAY CONTRACT BALANCE
    //
    // Returns Ether stored in the contract in Wei.
    // --------------------------------------------------------

    function getContractBalance()
        public
        view
        returns (uint256)
    {

        return address(this).balance;
    }

    // --------------------------------------------------------
    // FUNCTION 19: WITHDRAW CONTRACT ETHER
    //
    // Only owner can withdraw Ether received by
    // the contract through receive() or fallback().
    //
    // Uses call() and a reentrancy guard.
    // --------------------------------------------------------

    function withdrawEther(
        uint256 amount
    )
        public
        onlyOwner
        nonReentrant
    {

        require(
            amount > 0,
            "Invalid amount"
        );

        require(
            amount <= address(this).balance,
            "Insufficient balance"
        );

        // Transfer Ether to contract owner.
        (bool success, ) = payable(owner).call{
            value: amount
        }("");

        require(
            success,
            "Transfer failed"
        );

        emit EtherWithdrawn(
            owner,
            amount
        );
    }

}


// ============================================================
// REMIX IDE EXECUTION GUIDE
// ============================================================
//
// STEP 1: CREATE FILE
//
// Name: StudentData.sol
//
// Paste the complete code above.
//
// STEP 2: COMPILE
//
// Select Solidity compiler version 0.8.20.
// Click Compile StudentData.sol.
//
// Both contracts will be available in the contract dropdown.
//
// STEP 3: DEPLOY SIMPLE CONTRACT
//
// Select StudentData.
// Keep Value = 0 Wei.
// Click Deploy.
//
// STEP 4: TEST SIMPLE CONTRACT
//
// addStudent(101, "Shravan", 85)
// addStudent(102, "Rahul", 90)
// addStudent(103, "Neha", 78)
//
// getStudent(0)
// Expected: 101, Shravan, 85
//
// getCount()
// Expected: 3
//
// STEP 5: DEPLOY ADVANCED CONTRACT
//
// Select AdvancedStudentData.
// Deploy only once.
//
// STEP 6: ADD MULTIPLE STUDENTS
//
// In addMultipleStudents(), enter:
//
// rollNos:
// [101, 102, 103, 104]
//
// names:
// ["Shravan", "Rahul", "Neha", "Amit"]
//
// marks:
// [85, 90, 78, 35]
//
// Click Transact.
//
// STEP 7: TEST ADVANCED FUNCTIONS
//
// getCount()
// getAllStudents()
// getStudentByIndex(0)
// getStudentByRollNo(102)
//
// updateStudent(102, "Rahul", 95)
//
// deleteStudent(104)
//
// getAverageMarks()
// getTopper()
// getPassingCount()
// getFailingCount()
// getClassSummary()
//
// STEP 8: TEST RECEIVE FUNCTION
//
// Set Remix Value = 0.01 Ether.
// Send a transaction with empty calldata
// using the low-level interaction section.
//
// Expected:
// receiveCallCount increases by 1.
// Ether is transferred to the contract.
//
// STEP 9: TEST FALLBACK FUNCTION
//
// Set Value = 0.01 Ether.
// Open Low Level Interaction.
// Enter unknown calldata:
//
// 0x12345678
//
// Click Transact.
//
// Expected:
// fallbackCallCount increases by 1.
// FallbackExecuted event appears in logs.
//
// STEP 10: CHECK CONTRACT BALANCE
//
// getContractBalance()
//
// STEP 11: WITHDRAW ETHER
//
// Set Remix Value = 0 Wei.
// Enter the withdrawal amount in Wei:
//
// 10000000000000000
//
// This represents 0.01 Ether.
//
// Click withdrawEther().
//
// STEP 12: OBSERVE GAS
//
// Open Remix Terminal.
// Expand each transaction.
// Observe:
// - Gas Used
// - Transaction Cost
// - Effective Gas Price
//
// Transaction Fee = Gas Used * Effective Gas Price
//
// ============================================================
// SAMPLE RESULTS
// ============================================================
//
// Add these students:
//
// 101 - Shravan - 85
// 102 - Rahul   - 90
// 103 - Neha    - 78
// 104 - Amit    - 35
//
// Expected:
//
// Total Students = 4
// Total Marks = 288
// Average Marks = 72
// Passing Students = 3
// Failing Students = 1
//
// Topper:
// Roll No = 102
// Name = Rahul
// Marks = 90
//
// After deleting student 104:
//
// Total Students = 3
//
// NOTE:
// These are illustrative results.
// Solidity integer division removes the decimal part.
//
// ============================================================
// IMPORTANT EXTERNAL VIVA QUESTIONS
// ============================================================
//
// 1. What is a Smart Contract?
// A program deployed on a blockchain that executes
// according to predefined conditions.
//
// 2. What is Solidity?
// A high-level programming language for writing
// smart contracts on Ethereum-compatible blockchains.
//
// 3. What is a Structure?
// A user-defined data type that combines different
// variables into a single record.
//
// 4. What is an Array?
// A data structure that stores multiple values
// of the same type.
//
// 5. What is a Dynamic Array?
// An array whose size can increase or decrease.
//
// 6. What is a Fallback Function?
// A special function executed when a function selector
// does not match a declared function.
//
// 7. What is a Receive Function?
// A special function executed when Ether is received
// with empty calldata.
//
// 8. Difference between receive() and fallback():
// receive() handles empty calldata.
// fallback() handles unmatched function selectors.
// If receive() is absent, fallback() can also
// handle empty-calldata Ether transfers.
//
// 9. Why do we define both functions?
// To handle direct Ether transfers and unknown
// function calls separately.
//
// 10. What is a Mapping?
// A key-value data structure.
//
// 11. Why is Mapping used in the advanced contract?
// It provides direct average-time lookup of a student
// by roll number.
//
// 12. What is a Modifier?
// A reusable condition applied to functions.
//
// 13. What is onlyOwner?
// It restricts sensitive operations to the owner.
//
// 14. What is a Constructor?
// A function executed once during deployment.
//
// 15. What is msg.sender?
// The address calling the current function.
//
// 16. What is msg.value?
// The Ether amount sent with the current transaction.
//
// 17. What is msg.data?
// The calldata supplied with the transaction.
//
// 18. What is payable?
// It allows a function to receive Ether.
//
// 19. What is emit?
// It generates an event in transaction logs.
//
// 20. What is require()?
// It validates a condition and reverts if false.
//
// 21. What is Swap-and-Pop?
// It replaces a deleted array element with the last
// element and removes the last element.
//
// 22. Does Swap-and-Pop maintain array order?
// No, the order can change.
//
// 23. What is Gas?
// A measurement of computational work performed
// by the Ethereum Virtual Machine.
//
// 24. What is Gas Used?
// The amount of gas actually consumed by a transaction.
//
// 25. What is Gas Limit?
// The maximum gas allowed for a transaction.
//
// 26. What is Transaction Fee?
// Gas Used multiplied by Effective Gas Price.
//
// 27. Do view functions consume transaction fees?
// A normal external CALL to a view function does not
// create an on-chain transaction fee.
// A view function called from another transaction
// still uses gas within that transaction.
//
// 28. Why is addMultipleStudents useful?
// It inserts multiple records in one transaction,
// rather than submitting separate transactions.
//
// 29. What happens if one record is invalid in bulk insertion?
// The complete transaction reverts.
// Earlier changes made in that same transaction are undone.
//
// 30. Why use studentIndex mapping?
// It avoids searching through every array element
// to find a student's position.
//
// 31. What is the index + 1 technique?
// Mapping value zero represents a missing student.
// Valid array indexes are stored as index + 1.
//
// 32. What is the purpose of the reentrancy guard?
// It prevents a protected function from being
// entered again before its current execution completes.
//
// 33. What is the difference between storage and memory?
// Storage persists on the blockchain.
// Memory is temporary during function execution.
//
// 34. What is calldata?
// Read-only data supplied to an external function call.
//
// 35. What is an Event?
// A log emitted by the contract that external applications
// can observe.
//
// 36. Can private data be hidden on blockchain?
// No. Private only controls Solidity-level access.
// Blockchain state is publicly inspectable.
//
// 37. What happens if a student is not found?
// The require condition fails and the operation reverts.
//
// 38. What is the time complexity of getTopper()?
// O(n), because all student records may be checked.
//
// 39. What is the time complexity of mapping lookup?
// O(1) average-time lookup.
//
// 40. What is the difference between Remix VM and Sepolia?
// Remix VM is a local blockchain simulation.
// Sepolia is a public Ethereum test network.
//
// ============================================================
// COMPLEXITY ANALYSIS
// ============================================================
//
// Let n = Number of students.
//
// SIMPLE CONTRACT:
//
// addStudent()         : O(1) amortized
// getStudent()         : O(1)
// getCount()           : O(1)
//
// ADVANCED CONTRACT:
//
// addStudent()         : O(1) average / amortized
// addMultipleStudents(): O(n)
// getStudentByIndex()  : O(1)
// getStudentByRollNo() : O(1) average
// updateStudent()      : O(1) average
// deleteStudent()      : O(1) average
// getAllStudents()     : O(n)
// getAverageMarks()    : O(n)
// getTopper()          : O(n)
// getPassingCount()    : O(n)
// getFailingCount()    : O(n)
// getClassSummary()    : O(n)
//
// SPACE COMPLEXITY:
//
// Student array     : O(n)
// Student mapping   : O(n)
// Additional loops  : O(1) auxiliary space
//
// IMPORTANT:
// These are algorithmic estimates.
// Actual gas consumption depends on storage writes,
// dynamic strings, calldata size, compiler optimization
// and network gas rules.
//
// ============================================================
// FINAL PRACTICAL SUMMARY
// ============================================================
//
// BASIC VERSION:
// Structure + Array + Receive + Fallback.
//
// ADVANCED VERSION:
// Structure + Array + Mapping + Modifier +
// Constructor + Events + Receive + Fallback +
// Loops + Multiple Student Operations.
//
// GAS OBSERVATION:
// Compare the gas used by single insertion,
// bulk insertion, update and deletion.
//
// IMPORTANT EXTERNAL VIVA POINT:
//
// Structure stores student information.
// Array stores multiple student records.
// Mapping provides direct record lookup.
// Receive handles Ether with empty calldata.
// Fallback handles unknown function selectors.
// Gas measures the computational work of a transaction.
//
// ============================================================
// END OF PRACTICAL
// ============================================================
