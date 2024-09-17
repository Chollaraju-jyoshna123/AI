8 Queens problem:
Initialize the Board:
Create an empty 8x8 chessboard. Each cell can either be empty (0) or have a queen (1).
Start with the First Row:
Begin by placing a queen in the first row, trying each column (from 0 to 7).
Check if the Position is Safe:
For each column in the current row, check if placing a queen there is safe:
No queen in the same column.
No queen in the upper left diagonal.
No queen in the upper right diagonal.
Place the Queen:
If a safe position is found in the current row and column, place the queen there (set the value of that cell to 1).
Recur for the Next Row:
Move to the next row and repeat the process of trying to place a queen in one of the columns while checking for safety.
Backtrack if Necessary:
If no valid position is found for a queen in the current row (i.e., all columns lead to conflicts), backtrack:
Remove the queen from the previous row (unset the last placed queen by setting the cell back to 0).
Try placing the queen in a different column in the previous row.
Repeat Until All Queens are Placed:
Continue the process of placing queens and backtracking until a valid solution is found where all 8 queens are placed on the board without threatening each other.
Stop When a Solution is Found:
Once all 8 queens are successfully placed, the algorithm terminates. The board configuration represents a valid solution to the 8 Queens problem.
Explore All Solutions (Optional):
If needed, the algorithm can be modified to find all possible solutions by continuing the search even after finding a valid configuration.


Water jug:
1.Define States and Actions:
•	Each state can be defined as the amount of water in Jug 1 and Jug 2, i.e., (a, b) where a is the amount in Jug 1 and b is the amount in Jug 2.
•	The possible actions (state transitions) are:
1.	Fill Jug 1 (set a = x)
2.	Fill Jug 2 (set b = y)
3.	Empty Jug 1 (set a = 0)
4.	Empty Jug 2 (set b = 0)
5.	Pour water from Jug 1 to Jug 2 (Pour as much as possible, without exceeding the capacity of Jug 2)
6.	Pour water from Jug 2 to Jug 1 (Pour as much as possible, without exceeding the capacity of Jug 1)
2.Initialize the Search:
•	Start with both jugs empty: (0, 0).
•	Create a queue (for BFS) and add the initial state (0, 0) to it.
•	Maintain a set of visited states to avoid revisiting the same state.
3.Breadth-First Search (BFS) Process:
•	While the queue is not empty, do the following:
1.	Dequeue the front element from the queue. Let this be the current state (a, b).
2.	Check if this state equals the goal (z, _) or (_, z) where one of the jugs contains exactly z liters of water.
3.	If it does, the solution is found, and you can stop.
4.	Otherwise, generate all possible new states by applying each of the 6 possible actions to (a, b) (fill, empty, pour).
5.	For each new state, if it hasn’t been visited before, add it to the queue and mark it as visited.
4.Checking for a Solution:
•	If you find a state where either Jug 1 or Jug 2 contains exactly z liters of water, you have a solution.
•	If the queue is exhausted without finding a solution, it means it is impossible to measure z liters using the given jugs.


CyptoArithmetic:
1. Understand the Problem:
Each letter in the problem represents a unique digit (0-9).
The number represented by the letters must satisfy the given arithmetic operation.
For example, in the problem: SEND + MORE = MONEY
You need to determine which digit each letter (S, E, N, D, M, O, R, Y) represents.
2. Identify Unique Letters:
List all the unique letters involved in the problem.
In the example above, the unique letters are: S, E, N, D, M, O, R, Y.
3. Determine Constraints:
The first letter in a number cannot represent 0 (for example, S and M cannot be 0).
Some letters may have additional constraints based on the operation (such as possible carries).
4. Use Trial and Error (Backtracking):
Assign digits to letters and check if the assignment satisfies the equation.
Start by assigning digits to letters and calculating the resulting numbers.
Ensure that each letter has a unique digit.
5. Check Consistency:
After each digit assignment, check whether the partial equation still holds.
For example, when calculating SEND + MORE = MONEY, check the last digits (D + E = Y) first, and then proceed from right to left, ensuring each partial sum is correct, including any carry values.
6. Backtrack if Necessary:
If a contradiction is found (e.g., the sum doesn’t match or the same digit is assigned to two letters), backtrack by undoing the last assignment and trying a new digit.
This systematic approach is called backtracking, where you explore different digit combinations until a valid solution is found.
7. Optimize with Known Constraints:
Use constraints to reduce the number of digit assignments you need to try.
For example, in SEND + MORE = MONEY, since the sum is 5 digits long, M must be 1 (because the sum of two 4-digit numbers can only give a 5-digit number if the first digit is 1).
8. Repeat Until a Solution is Found:
Continue assigning digits and checking until you find the unique combination that satisfies the equation.
You can also verify the solution by plugging the assigned digits back into the original equation.


A star algorithm:
1.Initialization:
•	Create Open and Closed Sets:
o	Open Set: A priority queue or min-heap containing nodes to be evaluated, initialized with the start node.
o	Closed Set: A set of nodes that have already been evaluated.
•	Initialize Costs:
o	g_cost for each node: The cost from the start node to the current node.
o	f_cost for each node: The estimated total cost (g_cost + h_cost).
o	h_cost: The heuristic estimate from the current node to the goal node.
•	Set Start Node Costs:
o	g_cost of the start node to 0.
o	f_cost of the start node to its h_cost.
2.Algorithm Execution:
•	While the Open Set is not Empty:
1.	Select the Node with the Lowest f_cost:
	Dequeue the node with the lowest f_cost from the Open Set. This node is called current_node.
2.	Check for Goal:
	If current_node is the goal node, reconstruct the path from start to goal and return it.
3.	Move Node to Closed Set:
	Add current_node to the Closed Set.
4.	Evaluate Neighbors:
	For each neighbor of current_node:
	If the Neighbor is in the Closed Set:
	Skip it as it has already been evaluated.
	Calculate Costs:
	Calculate the tentative g_cost (i.e., g_cost of current_node plus the cost to reach the neighbor).
	If the neighbor is not in the Open Set or the new g_cost is lower than its current g_cost:
	Update the g_cost and f_cost of the neighbor.
	Set the current_node as the parent of the neighbor (for path reconstruction).
	If the neighbor is not in the Open Set, add it to the Open Set.
5.	Continue Until Path is Found or Open Set is Empty:
	If the Open Set becomes empty without finding the goal, report that there is no path.
3.Path Reconstruction:
•	Once the goal node is reached, reconstruct the path from the start node to the goal node by following parent pointers from the goal node back to the start node.


Depth First Search:
1.Initialization:
•	Create a Set for Visited Nodes: Initialize an empty set or list to keep track of visited nodes.
2.Recursive DFS Function:
•	Start from a Node:
o	Mark the current node as visited.
o	Process the current node.
o	For each unvisited neighbor of the current node, recursively call DFS on that neighbor.
3.Termination:
•	The recursion ends when all reachable nodes from the starting node have been visited.


Colour mapping
