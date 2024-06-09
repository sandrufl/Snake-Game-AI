import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        #2 linear layers: linear1 gets input size as input size and hidden size as output
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # it gets x so the tensor... we apply the linear error and use the activation function
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x) # we dont need activation function at the end
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        #we check if a file exists already if not we create one
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        #now we update the file
        file_name = os.path.join(model_folder_path, file_name)
        # we save the file only saving this state dictionary
        torch.save(self.state_dict(), file_name)

class QTrainer:
    # lr = learning rate
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # criterion for loss function nothing else than the mean squared error
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x) it appends one dimension in the beginning 
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with the current state
         # old: Q = model.predict(state0)
        pred = self.model(state)  # 3 different values (the action)

        target = pred.clone()
        for idx in range(len(done)):
            Qnew = reward[idx]
            if not done[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(next_state))
            
            target[idx][torch.argmax(action).item()] = Qnew

        # 2: part is the bellman simplified for q update rule
        # Qnew = R + gamma * max(Q(state1))
        # Qnew = r + gamma * max(next_predicted Q value) (only one value) (only do this if not done)
        #to get 3 values here aswell 
        #pred.clone()
        #preds[argmax(action)] = Qnew

        #zero grad to empty the gradient
        self.optimizer.zero_grad()

        #calculate loss
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
