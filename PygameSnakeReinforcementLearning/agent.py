import torch
import random
import numpy as np
from collections import deque #data structure to store memory
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)#if we excede max memory it will remove elements from the left (pop left)
        # first one is the size of the state (11 values) and output size is 3 because of the list of 3 
        # that will give us the direction. Hidden size can be different
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        # example state = [0,0,0 dang. straight, dang. right, dang. left, 0,1,0,0 move direction left, right, up, down, 0,1,0,1 food location left right up down]
        # in the example there is no danger and the direction is right, the food is right and down

        return np.array(state, dtype=int) # returns np.array of ints to also convert bools into 0s and 1s

    def remember(self, state, action, reward, next_state, done):
        #save all data as a big tuple
        self.memory.append((state, action, reward, next_state, done)) #popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        #take a batch from our memory to analize (sample) if we have a batch saved
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #list of tuples
        else: #if the saved data doesn't exceed batch size then take whole data
            mini_sample = self.memory
        
        #TODO: check out this zip function that extracts the data from the tuples in the mini sample
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        # in the beginning we want to do more random moves to explore the environment but later on we want to more exploit the agent/model
        # for this we use epsilon
        self.epsilon = 80 - self.n_games # hardcoded

        #the more games we have the less the int is gonna be less than epsilon so the less random moves
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            #not random but based on model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            #here the prediction is a tensor and could be of raw values like [5.0, 2.7, 0.1] so we do max to have [1, 0, 0]
            move = torch.argmax(prediction).item() #it gives a tensor so we make it an item
            final_move[move] = 1
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    game = SnakeGameAI()
    try:
        with open('record_and_games.txt', 'r') as file:
            lines = file.readlines()
            record = int(lines[0])
            agent.n_games = int(lines[1])
    except FileNotFoundError:
        # set record and games count to 0 if the file is not found
        record = 0
        agent.n_games = 0 

    model_folder_path = './model'
    if os.path.exists(model_folder_path):
        file_name = 'model.pth'
        file_name = os.path.join(model_folder_path, file_name)
        agent.model.load_state_dict(torch.load(file_name))
        agent.model.eval()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if(done):
            # train long memory (replay memory) (experienced memory) trains again on all the previous done moves
            # plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                with open('record_and_games.txt', 'w') as file:
                    file.write(str(record) + '\n')
                    file.write(str(agent.n_games))
                    
            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()