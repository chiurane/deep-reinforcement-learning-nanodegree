params = {'BUFFER_SIZE':int(1e6), # Replay buffer size
         'BATCH_SIZE':256, # minibatch size
         'GAMMA':0.99, # discount factor
         'TAU':1e-3, # for soft update of target parameters
         'LR_ACTOR':1e-4, # learning rate of the Actor
         'LR_CRITIC':1e-3, # Learning rate of the Critic
         'WEIGHT_DECAY':0., # L2 weight decay,
         'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}