
// ============================================================
// BLOCKCHAIN TECHNOLOGY PRACTICAL
// Experiment: Bank Account Smart Contract Using Solidity

// Objective (Question):
// Write a smart contract on a test network to perform
// the following bank account operations:
//
// 1. Deposit Money
// 2. Withdraw Money
// 3. Show Balance
//
// Programming Language: Solidity
// IDE: Remix IDE
// Compiler Version: Solidity 0.8.20
//
// HOW TO RUN:
// 1. Open https://remix.ethereum.org
// 2. Create a new file named BankAccount.sol
// 3. Paste this complete code.
// 4. Select Solidity Compiler 0.8.20.
// 5. Compile the contract.
// 6. Select Remix VM for local testing OR connect
//    a wallet configured with a supported test network.
// 7. Deploy the contract.
// 8. Test deposit, withdraw and balance functions.
//
// IMPORTANT:
// All monetary values are stored in Wei.
// 1 Ether = 10^18 Wei.
// In Remix, select Ether as the unit in the Value field.
// ============================================================


// ============================================================
// PART 1: SINGLE CUSTOMER BANK ACCOUNT
//
// Purpose:
// Create a bank account for one customer.
// The person deploying the contract becomes the customer.
//
// Operations:
// 1. Deposit money.
// 2. Withdraw money.
// 3. Show account balance.
//
// This implementation follows the structure of
// the reference code provided.
// ============================================================

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BankAccount {

    // Store the customer's wallet address
    address public customer;

    // Store the customer's bank balance
    uint256 private balance;

    // Constructor executes only once during deployment
    // The deployer's address becomes the customer
    constructor() {
        customer = msg.sender;
    }

    // --------------------------------------------------------
    // FUNCTION 1: DEPOSIT MONEY
    //
    // payable allows the function to receive Ether.
    // msg.value contains the amount sent in Wei.
    //
    // Any account can deposit into this single customer's
    // account, but only the customer can withdraw.
    // --------------------------------------------------------

    function deposit() public payable {

        require(
            msg.value > 0,
            "Amount must be greater than zero"
        );

        // Add deposited money to the balance
        balance += msg.value;
    }

    // --------------------------------------------------------
    // FUNCTION 2: WITHDRAW MONEY
    //
    // Only the customer who deployed the contract
    // is authorized to withdraw money.
    //
    // The amount parameter must be entered in Wei.
    // --------------------------------------------------------

    function withdraw(uint256 amount) public {

        // Check customer authorization
        require(
            msg.sender == customer,
            "Not authorized"
        );

        // Check valid withdrawal amount
        require(
            amount > 0,
            "Invalid amount"
        );

        // Check sufficient balance
        require(
            amount <= balance,
            "Insufficient balance"
        );

        // Reduce balance before transferring money
        balance -= amount;

        // Transfer Ether to the customer
        payable(customer).transfer(amount);
    }

    // --------------------------------------------------------
    // FUNCTION 3: SHOW BALANCE
    //
    // view means this function only reads data.
    // It does not modify blockchain state.
    // Returns balance in Wei.
    // --------------------------------------------------------

    function showBalance() public view returns (uint256) {

        return balance;
    }
}


// ============================================================
// PART 2: MULTIPLE CUSTOMER BANK ACCOUNT
//
// External Viva Extension:
//
// Question:
// What if the bank has multiple customers instead of one?
//
// Solution:
// Use Mapping to store separate balances for each
// customer's Ethereum wallet address.
//
// Features:
// 1. Multiple customers can deposit money.
// 2. Each customer can withdraw only their own money.
// 3. Display individual customer balance.
// 4. Display any customer's balance.
// 5. Display total number of customers.
// 6. Display all registered customer addresses.
// 7. Display total deposits.
// 8. Display total withdrawals.
// 9. Display total remaining bank balance.
// 10. Display actual Ether held by the contract.
//
// ============================================================

contract MultiCustomerBank {

    // Mapping stores each customer's individual balance.
    // Wallet Address => Balance in Wei
    mapping(address => uint256) private balances;

    // Track whether a customer has already registered
    mapping(address => bool) private registered;

    // Store customer wallet addresses for listing
    address[] private customerList;

    // Total money deposited throughout contract lifetime
    uint256 public totalDeposited;

    // Total money withdrawn throughout contract lifetime
    uint256 public totalWithdrawn;

    // Total current balance across all customer accounts
    uint256 public totalCustomerBalance;

    // --------------------------------------------------------
    // FUNCTION 1: DEPOSIT MONEY
    //
    // Every customer can deposit money into their own account.
    // msg.sender identifies the customer's wallet address.
    // --------------------------------------------------------

    function deposit() public payable {

        require(
            msg.value > 0,
            "Amount must be greater than zero"
        );

        // Register customer only on their first deposit
        if (!registered[msg.sender]) {

            registered[msg.sender] = true;

            customerList.push(msg.sender);
        }

        // Update individual customer balance
        balances[msg.sender] += msg.value;

        // Update aggregated banking information
        totalDeposited += msg.value;

        totalCustomerBalance += msg.value;
    }

    // --------------------------------------------------------
    // FUNCTION 2: WITHDRAW MONEY
    //
    // Each customer can withdraw only their own balance.
    // One customer cannot withdraw another customer's money.
    // --------------------------------------------------------

    function withdraw(uint256 amount) public {

        // Check valid amount
        require(
            amount > 0,
            "Invalid amount"
        );

        // Check customer's available balance
        require(
            amount <= balances[msg.sender],
            "Insufficient balance"
        );

        // Update individual balance first
        balances[msg.sender] -= amount;

        // Update aggregated banking information
        totalCustomerBalance -= amount;

        totalWithdrawn += amount;

        // Transfer money to the customer
        payable(msg.sender).transfer(amount);
    }

    // --------------------------------------------------------
    // FUNCTION 3: SHOW MY BALANCE
    //
    // Displays the balance of the currently connected
    // customer account in Remix IDE.
    // --------------------------------------------------------

    function showBalance() public view returns (uint256) {

        return balances[msg.sender];
    }

    // --------------------------------------------------------
    // FUNCTION 4: SHOW ANY CUSTOMER BALANCE
    //
    // Accept a wallet address and return its balance.
    // Useful for bank account management.
    //
    // --------------------------------------------------------

    function showCustomerBalance(
        address customerAddress
    ) public view returns (uint256) {

        return balances[customerAddress];
    }

    // --------------------------------------------------------
    // FUNCTION 5: TOTAL NUMBER OF CUSTOMERS
    //
    // Returns the number of unique customers
    // who have made at least one successful deposit.
    // --------------------------------------------------------

    function getTotalCustomers() public view returns (uint256) {

        return customerList.length;
    }

    // --------------------------------------------------------
    // FUNCTION 6: DISPLAY CUSTOMER ADDRESS
    //
    // Returns customer wallet address using its index.
    // Index starts from 0.
    // --------------------------------------------------------

    function getCustomerAddress(
        uint256 index
    ) public view returns (address) {

        require(
            index < customerList.length,
            "Invalid customer index"
        );

        return customerList[index];
    }

    // --------------------------------------------------------
    // FUNCTION 7: DISPLAY ALL CUSTOMER ADDRESSES
    //
    // Returns all registered customer wallet addresses.
    // --------------------------------------------------------

    function getAllCustomers()
        public
        view
        returns (address[] memory)
    {
        return customerList;
    }

    // --------------------------------------------------------
    // FUNCTION 8: SHOW AGGREGATED BANK SUMMARY
    //
    // Returns:
    // 1. Total Customers
    // 2. Total Deposited Amount
    // 3. Total Withdrawn Amount
    // 4. Total Remaining Customer Balance
    // 5. Actual Ether held by the smart contract
    //
    // All monetary values are returned in Wei.
    // --------------------------------------------------------

    function showBankSummary()
        public
        view
        returns (
            uint256 customers,
            uint256 deposited,
            uint256 withdrawn,
            uint256 remainingBalance,
            uint256 actualContractBalance
        )
    {
        return (
            customerList.length,
            totalDeposited,
            totalWithdrawn,
            totalCustomerBalance,
            address(this).balance
        );
    }
}


// ============================================================
// REMIX IDE EXECUTION GUIDE
// ============================================================
//
// PART A: SINGLE CUSTOMER CONTRACT
//
// 1. Compile BankAccount.
// 2. Deploy BankAccount.
// 3. The selected deployment account becomes customer.
// 4. Under the Value field, enter 1 and select Ether.
// 5. Click deposit().
// 6. Click showBalance().
// 7. For withdrawing 0.5 Ether, enter:
//    500000000000000000
//    in the withdraw() parameter.
// 8. Click withdraw().
// 9. Click showBalance() again.
//
// Example:
//
// Deposit = 1 Ether
// Withdraw = 0.5 Ether
// Remaining Balance = 0.5 Ether
//
// Note:
// Gas fees are separate from the deposited amount.
//
// ------------------------------------------------------------
//
// PART B: MULTIPLE CUSTOMER CONTRACT
//
// 1. Compile MultiCustomerBank.
// 2. Deploy it using Account 1.
// 3. Select Account 1 and deposit 2 Ether.
// 4. Select Account 2 and deposit 3 Ether.
// 5. Select Account 3 and deposit 1 Ether.
// 6. Click getTotalCustomers().
// 7. Click getAllCustomers().
// 8. Click showBalance() using each account.
// 9. Click showBankSummary() to see aggregated results.
//
// IMPORTANT:
// Deploy the multi-customer contract only once.
// Switch Remix accounts to simulate different customers.
// All accounts must interact with the SAME deployed contract.
//
// ============================================================
// SAMPLE EXECUTION: SINGLE CUSTOMER
// ============================================================
//
// Customer Account: Account 1
//
// Operation 1:
// Deposit = 1 Ether
//
// Operation 2:
// Show Balance = 1000000000000000000 Wei
//
// Operation 3:
// Withdraw = 500000000000000000 Wei
//
// Operation 4:
// Show Balance = 500000000000000000 Wei
//
// Remaining Balance = 0.5 Ether
//
// ============================================================
// SAMPLE EXECUTION: MULTIPLE CUSTOMERS
// ============================================================
//
// Customer 1 deposits = 2 Ether
// Customer 2 deposits = 3 Ether
// Customer 3 deposits = 1 Ether
//
// Total Customers = 3
//
// Total Deposited = 6 Ether
//
// Now Customer 2 withdraws 1 Ether.
//
// Total Withdrawn = 1 Ether
//
// Remaining Customer Balance = 5 Ether
//
// Customer-wise balances:
//
// Customer 1 = 2 Ether
// Customer 2 = 2 Ether
// Customer 3 = 1 Ether
//
// Aggregated Bank Summary:
//
// Total Customers         = 3
// Total Deposited         = 6 Ether
// Total Withdrawn         = 1 Ether
// Total Remaining Balance = 5 Ether
// Actual Contract Balance = 5 Ether
//
// NOTE:
// These are illustrative values, not live Remix results.
//
// ============================================================
// IMPORTANT THEORY AND EXTERNAL VIVA QUESTIONS
// ============================================================
//
// 1. What is Blockchain?
// Blockchain is a distributed digital ledger that
// stores transactions in linked blocks.
//
// 2. What is Ethereum?
// Ethereum is a blockchain platform that supports
// smart contracts and decentralized applications.
//
// 3. What is a Smart Contract?
// A smart contract is a program deployed on a blockchain
// that automatically executes according to predefined rules.
//
// 4. What is Solidity?
// Solidity is a high-level programming language
// used to write smart contracts on Ethereum-compatible
// blockchain networks.
//
// 5. What is Remix IDE?
// Remix is an online IDE used to write, compile,
// deploy and test Solidity smart contracts.
//
// 6. What is an EVM?
// Ethereum Virtual Machine.
// It executes smart contract bytecode on Ethereum-compatible
// blockchain networks.
//
// 7. What is msg.sender?
// It represents the address of the account or contract
// that is currently calling the function.
//
// 8. What is msg.value?
// It represents the amount of Ether sent with
// the current transaction, measured in Wei.
//
// 9. What is Wei?
// Wei is the smallest denomination of Ether.
//
// 1 Ether = 10^18 Wei.
//
// 10. What is the payable keyword?
// payable allows a function or address to receive Ether.
//
// 11. Why is deposit() payable?
// Because it accepts Ether from the customer.
//
// 12. What is the purpose of the constructor?
// It executes only once when the contract is deployed.
// Here, it stores the deployer's wallet address
// as the customer.
//
// 13. What is address in Solidity?
// It is a data type used to store a blockchain wallet
// or contract address.
//
// 14. What is uint256?
// It is an unsigned 256-bit integer.
// It stores non-negative numerical values.
//
// 15. What is require()?
// It checks a condition before executing further code.
// If the condition is false, the transaction reverts.
//
// 16. Why is require() used in withdraw()?
// To check authorization, positive withdrawal amount
// and sufficient account balance.
//
// 17. What is public?
// It allows a function or variable to be accessed externally.
//
// 18. What is private?
// It restricts direct access to a state variable
// or function from external contracts.
// It does not make blockchain data secret.
//
// 19. What is view?
// A view function reads contract state
// without modifying it.
//
// 20. What is transfer()?
// It transfers Ether to a payable address.
// It reverts the transaction if the transfer fails.
//
// 21. What is a Mapping?
// Mapping is a key-value data structure.
// Here, the wallet address is the key
// and the customer's balance is the value.
//
// 22. Why is Mapping used in the multiple customer contract?
// Because every customer requires an independent balance.
// Mapping allows balances to be stored separately
// using wallet addresses.
//
// 23. What is an Array?
// An array stores multiple values of the same data type.
// Here, customerList stores customer wallet addresses.
//
// 24. What is the difference between Array and Mapping?
// Array stores values using numerical indexes.
// Mapping stores values using keys.
//
// 25. What is msg.sender used for in the
// multiple customer contract?
// It identifies the current customer so that
// the correct account balance is updated.
//
// 26. Can Customer 1 withdraw Customer 2's money?
// No. The withdraw function uses msg.sender
// to access only the caller's balance.
//
// 27. What is the difference between the two contracts?
//
// Single Customer:
// - One customer address.
// - One common balance.
// - Only the customer can withdraw.
//
// Multiple Customers:
// - Multiple wallet addresses.
// - Separate balance for every customer.
// - Each customer withdraws their own balance.
// - Aggregated bank information is available.
//
// 28. What is Gas?
// Gas measures the computational work required
// to execute a transaction on the blockchain.
//
// 29. Why do we pay Gas Fees?
// Gas compensates the network for processing
// and executing blockchain transactions.
//
// 30. What is Deployment?
// Deployment is the process of publishing
// a smart contract to a blockchain network.
//
// 31. What is a Test Network?
// A test network is a blockchain environment
// used to test applications without using real
// mainnet cryptocurrency.
//
// 32. What is Remix VM?
// Remix VM is a local simulated blockchain
// environment provided by Remix IDE.
// It is useful for practical testing.
//
// 33. What is a Transaction?
// A transaction is an operation submitted to
// the blockchain that can modify its state.
//
// 34. What is a Smart Contract State?
// It represents the data stored in a smart contract,
// such as customer addresses and account balances.
//
// 35. What happens if the customer withdraws more
// than the available balance?
// The require condition fails and the transaction
// is reverted.
//
// 36. What happens if deposit amount is zero?
// The transaction is reverted because the amount
// must be greater than zero.
//
// 37. What is the difference between transfer()
// and send()?
// transfer() reverts on failure.
// send() returns false on failure and does not
// automatically revert the transaction.
//
// 38. What is address(this).balance?
// It returns the actual amount of Ether held
// by the currently executing contract.
//
// 39. What is an Event in Solidity?
// An event records information in transaction logs
// so external applications can monitor contract activity.
//
// 40. What is the purpose of totalDeposited?
// It stores the cumulative value of all successful
// customer deposits.
//
// 41. What is the purpose of totalWithdrawn?
// It stores the cumulative value of successful withdrawals.
//
// 42. What is totalCustomerBalance?
// It stores the combined current balance of all
// registered customer accounts.
//
// 43. Why do we use customerList?
// Mapping cannot directly provide a list of all keys.
// The array helps retrieve registered customer addresses.
//
// 44. What is the difference between totalDeposited
// and totalCustomerBalance?
// totalDeposited is the lifetime total of deposits.
// totalCustomerBalance is the current sum of account balances.
//
// 45. Can smart contracts access private variables directly?
// No external account or contract can directly access
// a private state variable through the Solidity interface.
// However, blockchain storage is publicly inspectable.
//
// 46. Can a smart contract run automatically without
// any transaction?
// No. Smart contract execution is triggered by
// transactions or calls from other contracts.
//
// 47. What is Reentrancy?
// Reentrancy occurs when an external call causes
// a contract function to be called again before
// the original execution finishes.
//
// 48. How does this code reduce withdrawal risk?
// It reduces the customer's balance before transferring
// Ether, following the Checks-Effects-Interactions pattern.
// The checks are performed first, state updates second,
// and the external transfer happens last.
//
// 49. What is the difference between Blockchain and
// Traditional Banking?
// Traditional banking relies on centralized institutions.
// Blockchain uses a distributed ledger and
// programmable smart contracts.
//
// 50. Is this a real banking application?
// No. It is an educational smart contract demonstrating
// deposit, withdrawal, account balances and customer
// account management. It does not implement KYC,
// banking regulations, account recovery or production
// financial safeguards.
//
// ============================================================
// IMPORTANT SECURITY AND PRACTICAL POINTS
// ============================================================
//
// 1. Never deploy this educational practical with real funds.
//
// 2. The single-customer contract follows the reference:
//    anyone can deposit into the single account,
//    but only the deployer can withdraw.
//
// 3. The multi-customer contract gives each address
//    its own account balance.
//
// 4. All deposits and withdrawals are performed
//    using Wei as the base unit.
//
// 5. A failed withdrawal reverts the state changes
//    made by that transaction.
//
// 6. The aggregate customer balance should normally
//    match the Ether held in the contract.
//    Forced Ether transfers can cause a difference.
//
// 7. Private variables are not confidential.
//    Blockchain storage can be publicly inspected.
//
// 8. This is a practical demonstration and not
//    a production-ready banking system.
//
// ============================================================
// END OF PRACTICAL
// ============================================================
