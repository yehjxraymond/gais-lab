from config import llm_open_router

# Let's get the LLM to help us write some python code
# For this exercise, solve the following programming challenge
#
# Write a python program for a todo list. 
# The program should receive input via the command line.
# The program should run in a loop until it is terminated, and in each loop, it should print out the current list of tasks.
# The user should be able to add a new line by adding + to the command line input.
# The user should be able to mark a task as done by adding x to the item number.
# The user should be able to remove a task by adding - to the item number.

def main():
    # Change the prompt below to print the solution to the lab problem
    response = llm_open_router.invoke()
    
    print(response.content)

if __name__ == "__main__":
    main()
