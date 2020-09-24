import torch
import torch.nn as nn
import torch.nn.functional as F

def build_hidden_layer(input_dim, hidden_layers):
    """Build hidden layer.
    Params
    ======
        input_dim (int): Dimension of hidden layer input
        hidden_layers (list(int)): Dimension of hidden layers
    """
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers)>1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden

class ActorCritic(nn.Module):
    def __init__(self, in_channel, hidden_channels, kernel_sizes, strides,
                    observation_space_h, observation_space_w, action_space,
                    shared_layers, critic_hidden_layers=[], actor_hidden_layers=[],
                    seed=0, init_type=None):
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_space))

        # 1st block
        self.conv1 = nn.Conv1d(in_channel, hidden_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        # 2nd block
        self.conv2 = nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        # 3rd block
        self.conv3 = nn.Conv1d(hidden_channels[1], hidden_channels[1], kernel_size=kernel_sizes[2], stride=strides[2])
        self.bn3 = nn.BatchNorm1d(hidden_channels[1])

        ######################################################
        # Manually calc number of Linear input connections
        # which depends on output of conv1d layers
        ######################################################
        def conv1d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv1d_size_out(conv1d_size_out(conv1d_size_out(observation_space_w,kernel_sizes[0],strides[0]),
                                                kernel_sizes[1],strides[1]),
                                                kernel_sizes[2],strides[2])
        convh = conv1d_size_out(conv1d_size_out(conv1d_size_out(observation_space_h,kernel_sizes[0],strides[0]),
                                                kernel_sizes[1],strides[1]),
                                                kernel_sizes[2],strides[2])

        #######################
        linear_input_size = convw * convh * hidden_channels[1]
        self.shared_layers = build_hidden_layer(input_dim=linear_input_size,
                                                hidden_layers=shared_layers)

        ######################################################
        # Add critic layers
        ######################################################
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                    hidden_layers=critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        ######################################################
        # Add actor layers
        ######################################################
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                   hidden_layers=actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_space)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_space)

        #######################

        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()

        # Initialize hidden and actor-critic layers
        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        """Initialize network weights.
        """
        if isinstance(n, nn.Linear):
            if self.init_type == 'xavier-uniform':
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type == 'xavier-normal':
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type == 'kaiming-uniform':
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type == 'kaiming-normal':
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type == 'orthogonal':
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type == 'uniform':
                nn.init.uniform_(n.weight.data)
            elif self.init_type == 'normal':
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError('initialization type is not found in the set of existing types')

    def forward(self, state):
        """Build a network that maps state -> (action, value)."""
        def apply_multi_layer(layers,x,f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = apply_multi_layer(self.shared_layers,state.view(state.size(0),-1))

        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden,v_hid)

        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden,a_hid)

        action = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return action, value