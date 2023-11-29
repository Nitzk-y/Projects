#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# In[3]:


def swish(x):
    return torch.sigmoid(x)*x


class Diffusion(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(Diffusion, self).__init__()

        self.layer1 = nn.Linear(input_dim,450)
        self.layer2 = nn.Linear(450,400)
        self.layer3 = nn.Linear(400,280)
        self.layer4 = nn.Linear(280,280)
        self.layer5 = nn.Linear(280,280)
        self.layer6 = nn.Linear(280,280)
        self.layer7 = nn.Linear(280,output_dim)

        self.activ1 = nn.CELU()
        self.activ2 = swish

    def forward(self, m_out):
        m_out = self.layer1(m_out)
        m_out = self.activ1(m_out)

        m_out = self.layer2(m_out)
        m_out = self.activ2(m_out)

        m_out = self.layer3(m_out)
        m_out = self.activ1(m_out)

        m_out = self.layer4(m_out)
        m_out = self.activ2(m_out)

        m_out = self.layer5(m_out)
        m_out = self.activ1(m_out)

        m_out = self.layer6(m_out)
        m_out = self.activ1(m_out)

        m_out = self.layer7(m_out)
        return m_out        

    def sample(self, batch_size: int, device: str) -> torch.Tensor:
        
        draws = torch.zeros(batch_size,input_size-1)
        for j in range(batch_size):
            X = torch.randn(1,input_size-1,dtype=torch.float).to(device)

            for t in torch.arange(len(alphas)-1,-1,-1).to(device):
                if t > 0:
                    z = (torch.zeros(1,input_size-1)).to(device)
                else:
                    z = (torch.randn(1,input_size-1)).to(device)

                m_out = self.forward(torch.cat((X,torch.reshape(t_steps[t:t+1],(1,1))), 1).to(device))
                    
                X = 1/torch.sqrt(alphas[t]) * (X - (1 - alphas[t])/torch.sqrt(1 - alphas[t]) * m_out) + torch.sqrt(betas[t]) * z
                draws[j,:] = X#(torch.exp(C_inv*(X/2)+M)-1)[0,:]

        return draws


# In[4]:


betas = torch.arange(0.0001,0.0035,0.00010)
alphas = torch.cumprod(1-betas,dim=0)
t_steps = torch.arange(-1,1+2/(len(betas)-1),2/(len(betas)-1))


# In[5]:


X_train = torch.tensor(np.random.randn(1500,2)).to(torch.float32)
X_test = torch.tensor(np.random.randn(250,2)).to(torch.float32)

Y_train = torch.tensor((np.array([-1,1])[np.random.randint(0,2,(1500,2))]*np.random.exponential(0.5,(1500,2)))).to(torch.float32)
Y_test = torch.tensor((np.array([-1,1])[np.random.randint(0,2,(1500,2))]*np.random.exponential(0.5,(1500,2)))).to(torch.float32)


# In[6]:


input_size = X_train.shape[1]+1  # Input features
output_size = Y_train.shape[1]  # Output classes
learning_rate = 0.00001#2*(10**0)
num_epochs = 1500


# In[7]:


m = Diffusion(input_size,output_size)


# In[12]:


loss_function = nn.MSELoss()
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)

input_data = X_train
targets = Y_train

batch_size = 500

for epoch in range(num_epochs):
    T = np.random.randint(len(betas))
    epsilon = np.random.randn(X_train.shape[0],X_train.shape[1])
    shuffled_inds = np.arange(X_train.shape[0])
    random.shuffle(shuffled_inds)
    for batch in range(1,int(np.ceil(X_train.shape[0]/batch_size))+1):
        if batch*batch_size > X_train.shape[0]-1:
        #if batch == int(np.ceil(X_train.shape[0]/batch_size)):
            #print((1+(batch-1)*batch_size),X_train.shape[0])
            var_sum = torch.sqrt(alphas[T]) * X_train[shuffled_inds[(1+(batch-1)*batch_size):],:] + torch.sqrt(1-alphas[T])*epsilon[shuffled_inds[(1+(batch-1)*batch_size):],:]
            var_sum = var_sum.to(torch.float)
            times = torch.full((np.shape(X_train[shuffled_inds[(1+(batch-1)*batch_size):],:])[0],1),t_steps[T])
            times = times.to(torch.float)

            x_in = torch.cat((var_sum,times),dim=1)
            y_in = torch.tensor(epsilon[shuffled_inds[(1+(batch-1)*batch_size):],:])
            
            outputs = m(x_in.to(torch.float))
            loss = loss_function(outputs, y_in.to(torch.float))

            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            #print((1+(batch-1)*batch_size),batch*batch_size)
            var_sum = torch.sqrt(alphas[T]) * X_train[shuffled_inds[(1+(batch-1)*batch_size):batch*batch_size],:] + torch.sqrt(1-alphas[T])*epsilon[shuffled_inds[(1+(batch-1)*batch_size):batch*batch_size],:]
            var_sum = var_sum.to(torch.float)
            times = torch.full((np.shape(X_train[shuffled_inds[(1+(batch-1)*batch_size):batch*batch_size],:])[0],1),t_steps[T])
            times = times.to(torch.float)
            
            x_in = torch.cat((var_sum,times),dim=1)
            y_in = torch.tensor(epsilon[shuffled_inds[(1+(batch-1)*batch_size):batch*batch_size],:])
            
            outputs = m(x_in.to(torch.float))
            loss = loss_function(outputs, y_in.to(torch.float))

            opt.zero_grad()
            loss.backward()
            opt.step()
            
    #outputs = m(input_data)
    #loss = loss_function(outputs, targets)

    #opt.zero_grad()
    #loss.backward()
    #opt.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


print('Training finished.')


# In[9]:


samples = m.sample(300,"cpu").detach()


# In[10]:


plt.figure(figsize=(11,7))
plt.scatter(samples[:,0],samples[:,1])
plt.scatter(Y_train[:,0],Y_train[:,1])
plt.show()


# In[11]:


plt.figure(figsize=(11,7))
plt.scatter(samples[:,0],samples[:,1])
plt.scatter(Y_test[:,0],Y_train[:,1])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




