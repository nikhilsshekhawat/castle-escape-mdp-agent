import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
	   configuration and taking actions until a terminal state is reached.
	2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
	3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
	4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

	Some important notes:
		
		- The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
		
		- To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		  For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		  will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		- Your Q-table should be a dictionary with the following format:

				- Each key is a number representing the state (hashed using the provided hash() function), and each value should be an np.array
				  of length equal to the number of actions (initialized to all zeros).

				- This will allow you to look up Q(s,a) as Q_table[state][action], as well as directly use efficient numpy operators
				  when considering all actions from a given state, such as np.argmax(Q_table[state]) within your Bellman equation updates.

				- The autograder also assumes this format, so please ensure you format your code accordingly.
  
		  Please do not change this representation of the Q-table.
		
		- The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

		- Don't forget to reset the environment to the initial configuration after each episode by calling:
		  obs, reward, done, info = env.reset()

		- The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

		- The value of epsilon is initialized to 1. You are free to choose the decay rate.
		  No default value is specified for the decay rate, experiment with different values to find what works.

		- To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		  Example usage below. This function should be called after every action.
		  if gui_flag:
		      refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    
    # Track number of updates for each (state, action) pair
    num_updates = np.zeros((375, 6))  # 375 states, 6 actions
    
    for episode in range(num_episodes):
        # Reset environment
        obs, _, _, _ = env.reset()
        state = hash(obs)
        done = False
        
        while not done:
            # Initialize Q-values for new state if not seen before
            if state not in Q_table:
                Q_table[state] = np.zeros(6)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random action
                action = env.action_space.sample()
            else:
                # Exploit: choose best action
                action = np.argmax(Q_table[state])
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            next_state = hash(next_obs)
            
            if gui_flag:
                refresh(next_obs, reward, done, info)
            
            # Initialize Q-values for next state if not seen before
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(6)
            
            # Calculate learning rate eta
            eta = 1 / (1 + num_updates[state, action])
            
            # Calculate V(s') = max_a' Q(s', a')
            V_next = np.max(Q_table[next_state])
            
            # Q-learning update
            old_Q = Q_table[state][action]
            Q_table[state][action] = (1 - eta) * old_Q + eta * (reward + gamma * V_next)
            
            # Increment update counter
            num_updates[state, action] += 1
            
            # Move to next state
            state = next_state
        
        # Decay epsilon after each episode
        epsilon = epsilon * decay_rate
    
    return Q_table

# Set decay rate
decay_rate = 0.9999

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate)

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action = np.argmax(Q_table[state])
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment


