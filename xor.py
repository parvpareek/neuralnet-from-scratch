import torch
import random
from torch import nn as nn

class Xor(nn.Module):
    def __init__(self) -> None:
        super(Xor, self).__init__()
        
        self.l1 = nn.Linear(2, 2, bias=True)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(2,1, bias=True)
        
        
    def forward(self, x):
        #print("input")
        #print(x)
        x = self.l1(x)
        #print("L1")
        #print(x)
        x = self.relu(x)
        #print("Relu")
        #print(x)
        return self.l2(x)
    
    
def construct_dataset(size):
    
    data = torch.randint(0, 2, (size, 2)).float()
    label = (data[:, 0] != data[:, 1]).unsqueeze(-1).float()
    return data, label


if __name__ == '__main__':
    
    dataset, labels = construct_dataset(100)
    
    model = Xor()
    output = model(dataset)
    
    obj = nn.MSELoss()
    
    loss = obj(output, labels)
    
    print(loss)
    print(torch.cat((output, labels), dim=-1))
    print("L1")
    print(model.l1.weight.data)
    print("L2")
    print(model.l2.weight.data)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    
    
    num_steps = 10000
    for step in range(num_steps):
        # Forward pass
        output = model(dataset)
        
        # Compute loss
        loss = obj(output, labels)
        
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss and weights every 100 steps for monitoring
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
            x = input()
    
    # Final outputs after training
    output = model(dataset)
    loss = obj(output, labels)
    print(f"Final Output:\n{output}")
    print(f"Final Loss: {loss.item()}")
    
    print("L1")
    print(model.l1.weight.data)
    print("L2")
    print(model.l2.weight.data)

