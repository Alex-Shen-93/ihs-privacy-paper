#@title Class Definitions & Imports
import torch 
import torch.nn.functional as F
from enum import Enum
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def my_device():
  return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def linf_clipping(input, threshold):
  DEVICE = my_device()
  sign = torch.sign(input).to(torch.int8)
  output = torch.abs(input)

  return torch.clamp(output, min=torch.tensor(0).to(DEVICE), max=threshold) * sign

def modify_gradients(gradient_stack, defense_scenario, parameter_index, clip_is_known, normalize, for_training):
  DEVICE = my_device()
  torch.manual_seed(defense_scenario[DEFENSE_PARAMS.TORCH_SEED])
  original_shape = gradient_stack.shape
  new_gradients = gradient_stack.to(DEVICE).view((-1, gradient_stack.shape[-1]))

  s = 0
  for defense_strategy in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST]:
    if defense_strategy == DEFENSE_STRATEGY.NOISE_CLIP:
      noise, clip = defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][s][parameter_index]
      if not clip_is_known:
        clip = 999999

      if defense_scenario[DEFENSE_PARAMS.LINF_CLIP]:
        new_gradients = linf_clipping(new_gradients, clip)
      else:
        new_gradients = new_gradients/torch.unsqueeze(torch.maximum(torch.linalg.norm(new_gradients, ord=2, dim=-1)/clip, torch.tensor(1)), dim=-1)

      if noise > 0:
        if defense_scenario[DEFENSE_PARAMS.MAINTAIN_GRADIENT_SIGN]:
          sign = torch.sign(new_gradients).to(torch.int8)
          noise_tensor = torch.normal(torch.zeros(new_gradients.shape), noise*clip).to(DEVICE)
          new_gradients = torch.abs(new_gradients)
          new_gradients = torch.maximum(new_gradients + noise_tensor, torch.tensor(0))
          del noise_tensor
          new_gradients *= sign
          del sign
        else:
          noise_tensor = torch.normal(torch.zeros(new_gradients.shape), noise*clip).to(DEVICE)
          new_gradients += noise_tensor
          del noise_tensor

    elif defense_strategy == DEFENSE_STRATEGY.SCALE_NOISE:
      scale_noise = defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][s][parameter_index]
      if scale_noise > 0:
        scale = torch.normal(torch.zeros(new_gradients.shape[0]), scale_noise)
        positives = torch.maximum(torch.sign(scale), torch.tensor(0))
        new_gradients *= torch.unsqueeze((scale + 1) * positives + (1/(1 - scale)) * (1 - positives), dim=-1)
        del scale
        del positives

    elif defense_strategy == DEFENSE_STRATEGY.RANDOM_LINF:
      scale_noise = defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][s][parameter_index]
      linf_scale = 1 / (1 + torch.abs(torch.normal(torch.zeros(new_gradients.shape), scale_noise))).to(DEVICE)
      new_gradients = linf_clipping(new_gradients, torch.unsqueeze(torch.max(new_gradients, dim=-1)[0], dim=-1) * linf_scale) * 1 / linf_scale
      del linf_scale

    elif defense_strategy == DEFENSE_STRATEGY.NOISY_NOISE:
      noise, clip = defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][s][parameter_index]
      if not clip_is_known:
        clip = 999999

      if defense_scenario[DEFENSE_PARAMS.LINF_CLIP]:
        new_gradients = linf_clipping(new_gradients, clip)
      else:
        # new_gradients = new_gradients/torch.unsqueeze(torch.maximum(torch.linalg.norm(new_gradients, ord=2, dim=-1)/clip, torch.tensor(1)), dim=-1)
        pass 

      if noise > 0:
        # noise_tensor = torch.normal(torch.zeros(new_gradients.shape), noise*clip).to(DEVICE)
        # new_gradients += noise_tensor
        sign = torch.sign(new_gradients).to(torch.int8)
        # noise_noise_tensor = torch.abs(torch.normal(torch.zeros((new_gradients.shape[0], 1)), noise))
        noise_noise_tensor = torch.randint(0, 2, size=(new_gradients.shape[0], 1))

        noise_tensor = torch.normal(torch.zeros(new_gradients.shape), noise*clip*noise_noise_tensor*2).to(DEVICE)
        # noise_tensor = (torch.randint(0, 2, size=new_gradients.shape, device=DEVICE) - 0.5) * 2 * noise * clip
        del noise_noise_tensor

        new_gradients = torch.abs(new_gradients)
        new_gradients = torch.maximum(new_gradients + noise_tensor, torch.tensor(0))
        # new_gradients += noise_tensor
        del noise_tensor
        new_gradients *= sign
        del sign


    elif defense_strategy == DEFENSE_STRATEGY.GRADIENT_CENSOR:
      censor_rate = defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_PARAMS][s][parameter_index]
      if censor_rate > 0:
        if not defense_scenario[DEFENSE_PARAMS.CENSOR_SINGLE_TARGET] or for_training:
          print("Multi Target Censor")
          censor_vector = torch.rand(new_gradients.shape, device=DEVICE) > censor_rate
        else:
          print("Single Target Censor")
          censor_vector = torch.rand(new_gradients.shape[-1], device=DEVICE) > censor_rate
        new_gradients *= censor_vector
    else:
      raise NotImplementedError("Defense Strategy " + str(defense_strategy) + " not implemented")


    s += 1
  # END for defense_strategy in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST]:

  if normalize:
    new_gradients = new_gradients/torch.unsqueeze(torch.linalg.norm(new_gradients, ord=2, dim=-1), dim=-1)

  return new_gradients.view(original_shape)


class DEFENSE_STRATEGY(Enum):
  NOISE_CLIP = 0
  SCALE_NOISE = 1
  RANDOM_LINF = 2
  NOISY_NOISE = 3
  GRADIENT_CENSOR = 4


class DATA_PARAMS(Enum):
  NP_SEED = 0
  MICE_ITERATIONS = 1
  MIN_OBS_PER_CLIENT = 2
  IS_BINARY = 3


class TARGET_PARAMS(Enum):
  TORCH_SEED = 0
  EPOCHS = 1
  BATCH_SIZE = 2
  LEARNING_RATE = 3
  DROPOUT = 4
  FIRST_LAYER_FEATURES = 5
  LAYER_SCALINGS = 6


class FL_PARAMS(Enum):
  TORCH_SEED = 1
  NP_SEED = 2
  GRADIENT_UPDATES = 3
  CLIENT_PROP_PER_ROUND = 4
  LEARNING_RATE = 5
  DATA_MULTIPLIER = 6


class DEFENSE_PARAMS(Enum):
  TORCH_SEED = 0
  DEFENSE_STRATEGY_LIST = 1
  DEFENSE_STRATEGY_PARAMS = 2
  SCALE_GRADIENT_BY_SAMPLE_SIZE = 3
  LINF_CLIP = 4
  MAINTAIN_GRADIENT_SIGN = 5
  CENSOR_SINGLE_TARGET = 6


class ATTACK_PARAMS(Enum):
  TORCH_SEED = 0
  NP_SEED = 15
  DATA_ACCESS_MULTI = 16
  CLIP_IS_KNOWN = 1
  NORMALIZE_GRADIENTS = 2
  SEGMENTS = 3
  FOLDS = 4
  MAX_TRAIN_EPOCHS = 5
  BATCH_SIZE = 6
  LEARNING_RATE = 7
  DROPOUT = 8
  WEIGHT_DECAY = 9
  TEST_SAMPLE_MULTI = 10
  NOISE_IS_KNOWN = 11
  TRAIN_SAMPLE_MULTI = 12
  FIRST_LAYER_FEATURES = 13
  LAYER_SCALINGS = 14
  TEST_REPEATS = 17



class IHSDataset(torch.utils.data.Dataset):
  def __init__(self, data, labels=None, weights=None):
    self.data, self.labels, self.weights = data, labels, weights

  def __len__(self):
      return self.data.shape[0]

  def __getitem__(self, index):
    if self.labels is None:
      return {'data': self.data[index] }
    elif self.weights is None:
      return {'data': self.data[index], 'labels': self.labels[index] }
    else:
      return {'data': self.data[index], 'labels': self.labels[index], 'weights': self.weights[index] }


# --- General Model
class IHSNN(torch.nn.Module):
  def __init__(self, is_binary, num_inputs, first_layer_features, layer_scalings, dropout, use_batchnorm=False):
    super(IHSNN, self).__init__()
    self.is_binary = is_binary
    self.use_batchnorm = use_batchnorm

    self.network = self.create_layer(num_inputs, first_layer_features, dropout)
    next_layer = first_layer_features
    for scale in layer_scalings:
      self.network.extend(self.create_layer(
          next_layer,
          int(next_layer/scale),
          dropout
      ))
      next_layer = int(next_layer/scale)

    self.network.append(
        torch.nn.Linear(in_features=next_layer, out_features=1)
    )

    self.network=torch.nn.Sequential(*self.network)

  def create_layer(self, in_features, out_features, dropout):
    if self.use_batchnorm:
      return torch.nn.ModuleList([
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.BatchNorm1d(out_features),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(0.01)
      ])
    else:
      return torch.nn.ModuleList([
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(0.01)
      ])

  def create_optimizer(self, learning_rate, weight_decay=0):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

  def forward(self, data):
    if self.is_binary:
      logits = self.network(data)
      return logits, torch.sigmoid(logits)
    else:
      return self.network(data)

  def predict(self, data):
    if self.is_binary:
      logits, _ = self(data)
      return (logits > 0).float()
    else:
      return None

  def compute_loss_acc(self, data, labels, weights=None):
    self.eval()
    with torch.no_grad():
      if self.is_binary:
        loss = F.binary_cross_entropy(self(data)[1], labels, weight=weights).item()
        acc = roc_auc_score(labels.cpu(), self.predict(data).cpu())
        cm = confusion_matrix(labels.cpu(), self.predict(data).cpu())
      else:
        predictions = self(data)
        loss = F.mse_loss(predictions, labels).item()
        acc = r2_score(labels.cpu(), predictions.cpu())
        cm = None 
    return loss, acc, cm
  
  def train_batch(self, batch):
    self.train()
    self.optimizer.zero_grad()
    if self.is_binary:
      logits, probs = self.forward(batch['data'])
      cost = F.binary_cross_entropy(probs, batch['labels'], weight=batch['weights'])
    else:
      predictions = self.forward(batch['data'])
      cost = F.mse_loss(predictions, batch['labels'])
    cost.backward()
    self.optimizer.step()

  def train_batch_federated(self, client_dict_list, defense_scenario, parameter_index, no_protection_flag):
    DEVICE = my_device()

   
    running_gradient_stack = self.get_gradient_stack(client_dict_list, repeats=1)
    if not no_protection_flag:
      running_gradient_stack = modify_gradients(
          running_gradient_stack, 
          defense_scenario, 
          parameter_index,
          clip_is_known=True,
          normalize=False,
          for_training=True
      ).to(DEVICE)

    if defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
      sample_sizes = torch.tensor([[client_dict['labels'].shape[0]] for client_dict in client_dict_list], device=DEVICE)
      running_gradient_stack *= sample_sizes
    
    running_gradients = running_gradient_stack.view((-1, running_gradient_stack.shape[-1]))

    self.train()
    self.zero_grad()

    # Take average of client gradient updates
    if DEFENSE_STRATEGY.GRADIENT_CENSOR in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST]:
      running_gradients = running_gradients.sum(dim=0) / (running_gradients != 0).sum(dim=0)
      running_gradients = torch.nan_to_num(running_gradients, 0)
    else:
      running_gradients = torch.mean(running_gradients, dim=0)
    
    # Put the averaged gradient back into the model parameters so that the optimizer can step
    i = 0
    for p in self.parameters():
      start = i
      i += torch.numel(p.grad)
      p.grad = running_gradients[start:i].reshape(p.grad.shape)
    
    assert i == torch.numel(running_gradients)
    self.optimizer.step()

  def get_gradient_stack(self, client_tuples_list, repeats):
    DEVICE = my_device()
    running_gradients = []
    self.train()
    for client_batch in client_tuples_list * repeats:
      self.zero_grad()
      if self.is_binary:
        logits, probs = self.forward(client_batch['data'])
        cost = F.binary_cross_entropy(probs, client_batch['labels'], weight=client_batch['weights'])
      else:
        predictions = self.forward(client_batch['data'])
        cost = F.mse_loss(predictions, client_batch['labels'])
      cost.backward()

      # Assemble all gradient components into a single vector
      total_grad = []
      for p in self.parameters():
        total_grad += p.grad.flatten().tolist()
      total_grad = torch.unsqueeze(torch.tensor(total_grad), dim=0).to(DEVICE)
      running_gradients.append(total_grad)
    return torch.cat(running_gradients).view((repeats, -1, running_gradients[0].shape[-1]))

# --- WESAD Model
class WESADNN(torch.nn.Module):
  def __init__(self, is_binary, num_inputs, first_layer_features, layer_scalings, dropout, use_batchnorm=False):
    super(WESADNN, self).__init__()
    self.is_binary = is_binary
    self.use_batchnorm = use_batchnorm

    self.network = self.create_layer(num_inputs, first_layer_features, dropout)
    next_layer = first_layer_features
    for scale in layer_scalings:
      self.network.extend(self.create_layer(
          next_layer,
          int(next_layer/scale),
          dropout
      ))
      next_layer = int(next_layer/scale)

    self.network.append(
        torch.nn.Linear(in_features=next_layer, out_features=3)
    )

    self.network=torch.nn.Sequential(*self.network)

  def create_layer(self, in_features, out_features, dropout):
    if self.use_batchnorm:
      return torch.nn.ModuleList([
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.BatchNorm1d(out_features),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(0.01)
      ])
    else:
      return torch.nn.ModuleList([
        torch.nn.Linear(in_features=in_features, out_features=out_features),
        torch.nn.Dropout(dropout),
        torch.nn.LeakyReLU(0.01)
      ])

  def create_optimizer(self, learning_rate, weight_decay=0):
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

  def forward(self, data):
    logits = self.network(data)
    return logits, torch.softmax(logits, dim=-1)

  def predict(self, data):
    logits = self.network(data)
    return torch.argmax(logits, dim=-1)

  def compute_loss_acc(self, data, labels, weights=None, get_cm=False):
    self.eval()
    with torch.no_grad():
      loss = F.cross_entropy(self.network(data), labels).item()
      acc = (labels == self.predict(data)).float().mean().item()
      cm = confusion_matrix(labels.cpu(), self.predict(data).cpu()) if get_cm else None
    return loss, acc, cm
  
  def train_batch(self, batch):
    self.train()
    self.optimizer.zero_grad()
    logits, probs = self.forward(batch['data'])
    cost = F.cross_entropy(logits, batch['labels'])
    cost.backward()
    self.optimizer.step()

  def train_batch_federated(self, client_dict_list, defense_scenario, parameter_index, no_protection_flag):
    DEVICE = my_device()
    running_gradient_stack = self.get_gradient_stack(client_dict_list, repeats=1)
    if not no_protection_flag:
      running_gradient_stack = modify_gradients(
          running_gradient_stack, 
          defense_scenario, 
          parameter_index,
          clip_is_known=True,
          normalize=False,
          for_training=True
      ).to(DEVICE)

    if defense_scenario[DEFENSE_PARAMS.SCALE_GRADIENT_BY_SAMPLE_SIZE]:
      sample_sizes = torch.tensor([[client_dict['labels'].shape[0]] for client_dict in client_dict_list], device=DEVICE)
      running_gradient_stack *= sample_sizes
    
    running_gradients = running_gradient_stack.view((-1, running_gradient_stack.shape[-1]))

    self.train()
    self.zero_grad()

    # Take average of client gradient updates
    if DEFENSE_STRATEGY.GRADIENT_CENSOR in defense_scenario[DEFENSE_PARAMS.DEFENSE_STRATEGY_LIST]:
      running_gradients = running_gradients.sum(dim=0) / (running_gradients != 0).sum(dim=0)
      running_gradients = torch.nan_to_num(running_gradients, 0)
    else:
      running_gradients = torch.mean(running_gradients, dim=0)
    
    # Put the averaged gradient back into the model parameters so that the optimizer can step
    i = 0
    for p in self.parameters():
      start = i
      i += torch.numel(p.grad)
      p.grad = running_gradients[start:i].reshape(p.grad.shape)
    
    assert i == torch.numel(running_gradients)
    self.optimizer.step()

  def get_gradient_stack(self, client_tuples_list, repeats):
    DEVICE = my_device()
    running_gradients = []
    self.train()
    for client_batch in client_tuples_list * repeats:
      self.zero_grad()
      logits, probs = self.forward(client_batch['data'])
      cost = F.cross_entropy(logits, client_batch['labels'])
      cost.backward()

      # Assemble all gradient components into a single vector
      total_grad = []
      for p in self.parameters():
        total_grad += p.grad.flatten().tolist()
      total_grad = torch.unsqueeze(torch.tensor(total_grad), dim=0).to(DEVICE)
      running_gradients.append(total_grad)
    return torch.cat(running_gradients).view((repeats, -1, running_gradients[0].shape[-1]))


# --- Gradient
class GradientDataset(torch.utils.data.Dataset):
  def __init__(self, gradients, labels=None, weights=None):
    self.gradients = gradients
    self.labels = labels
    self.weights = weights

  def __len__(self):
    return self.gradients.shape[0]

  def __getitem__(self, index):
    if self.labels is None:
      return {'data': self.gradients[index]}
    else:
      return {'data': self.gradients[index], 'labels': self.labels[index], 'weights': self.weights[index]}

