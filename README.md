# WWTPs-Code
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# --- TLSTM Model with Spatial Attention ---
class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        scores = torch.matmul(x, self.weight) + self.bias
        attention_weights = self.softmax(scores)
        return x * attention_weights

class TLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        
    def forward(self, x, h_c):
        return self.lstm_cell(x, h_c)


class TLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, attention=True):
        super(TLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([TLSTMCell(input_dim, hidden_dim) for _ in range(num_layers)])
        self.attention = SpatialAttention(input_dim) if attention else None

    def forward(self, x):
        if self.attention:
            x = self.attention(x)
        h_c = [(torch.zeros(x.size(0), self.hidden_dim), torch.zeros(x.size(0), self.hidden_dim)) for _ in range(self.num_layers)]
        outputs = []
        for t in range(x.size(1)):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h_c[i] = cell(input_t, h_c[i])
                input_t = h_c[i][0]
            outputs.append(input_t)
        return torch.stack(outputs, dim=1)

class RL:
    def __init__(self, input_dim, hidden_dim, num_layers, action_dim, lr=0.001, epsilon=0.2, gamma=0.99, beta=0.01):
        """
        RL Agent using TLSTM for state processing and MLP for policy and value estimation.

        :param input_dim: Input feature dimension.
        :param hidden_dim: Hidden layer size in TLSTM.
        :param num_layers: Number of TLSTM layers.
        :param action_dim: Number of actions (output dimension).
        :param lr: Learning rate.
        :param epsilon: Clipping range for updates.
        :param gamma: Discount factor for rewards.
        :param beta: Entropy coefficient to encourage exploration.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta

        # TLSTM for processing state sequences
        self.tlstm = TLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, attention=True)

        # MLP for policy and value output
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Optimizers
        self.optimizer = optim.Adam(
            list(self.tlstm.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

        # Replay buffer
        self.replay_buffer = []

    def store_transition(self, state, action, reward, next_state, old_prob):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, old_prob))

    def select_action(self, state_sequence):
        """
        Select an action based on the current policy.

        :param state_sequence: Input state sequence for TLSTM.
        :return: Selected action and its probability.
        """
        state_tensor = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        tlstm_output = self.tlstm(state_tensor)[:, -1, :]  # Use the last time step's output
        probs = self.policy_net(tlstm_output)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0, action].item()

    def compute_returns(self, rewards, dones, next_value):
        """
        Compute discounted returns for rewards.

        :param rewards: List of rewards.
        :param dones: List of done flags.
        :param next_value: Value of the next state.
        :return: Discounted returns.
        """
        returns = []
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, batch_size=32):
        """
        Perform an update using data from the replay buffer.

        :param batch_size: Number of samples per batch.
        """
        if len(self.replay_buffer) < batch_size:
            return  # Not enough data to train

        # Sample a batch of transitions
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, old_probs = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).unsqueeze(1)

        # Process states through TLSTM
        state_outputs = self.tlstm(states)[:, -1, :]  # Last time step output
        next_state_outputs = self.tlstm(next_states)[:, -1, :]

        # Compute targets for value network
        with torch.no_grad():
            next_values = self.value_net(next_state_outputs)
            targets = rewards + self.gamma * next_values

        # Compute advantages
        values = self.value_net(state_outputs)
        advantages = (targets - values).detach()

        # Compute new probabilities
        probs = self.policy_net(state_outputs).gather(1, actions)
        ratios = probs / old_probs

        # Clipped surrogate objective
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        # Value loss
        value_loss = nn.MSELoss()(values, targets)

        # Entropy loss (encourages exploration)
        entropy_loss = -torch.mean(probs * torch.log(probs + 1e-10))

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss - self.beta * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def clear_replay_buffer(self):
        """
        Clear the replay buffer to prepare for new data collection.
        """
        self.replay_buffer = []

class ABC:
    def __init__(self, objective_func, dim, bounds, population_size=30, max_iter=100):
        self.obj_func = objective_func
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])
        self.limit = np.zeros(population_size)
        self.best_solution = None
        self.best_fitness = np.inf

    def optimize(self):
        for _ in range(self.max_iter):
            # Employed bees
            for i in range(self.population_size):
                new_solution = self._mutate(i)
                new_fitness = self.obj_func(new_solution)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.limit[i] = 0
                else:
                    self.limit[i] += 1
            
            # Onlooker bees
            prob = self.fitness / np.sum(self.fitness)
            for _ in range(self.population_size):
                i = np.random.choice(range(self.population_size), p=prob)
                new_solution = self._mutate(i)
                new_fitness = self.obj_func(new_solution)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness

            # Scout bees
            for i in range(self.population_size):
                if self.limit[i] > 10:
                    self.population[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    self.fitness[i] = self.obj_func(self.population[i])
                    self.limit[i] = 0
            
            # Update best solution
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.population[best_idx]
                self.best_fitness = self.fitness[best_idx]

        return self.best_solution, self.best_fitness

    def _mutate(self, index):
        partner_idx = np.random.choice([i for i in range(self.population_size) if i != index])
        phi = np.random.uniform(-1, 1, self.dim)
        return np.clip(self.population[index] + phi * (self.population[index] - self.population[partner_idx]), self.bounds[0], self.bounds[1])


# --- Main Model Class for Transportation Mode Detection ---
class AnomalyDetector:
    def __init__(self, input_dim, hidden_dim, num_layers, attention=True):
        self.model = TLSTM(input_dim, hidden_dim, num_layers, attention)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)

            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = self.model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
        return predictions.numpy()

class DatasetHandler:
    def __init__(self, file_path, target_column, test_size=0.2):
        """
        Initialize the dataset handler.

        :param file_path: Path to the dataset file.
        :param target_column: The name of the target column.
        :param test_size: The proportion of the dataset to include in the test split.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def load_and_process_data(self):
        """
        Load and preprocess the dataset.
        
        :return: Processed train and test splits (X_train, X_test, y_train, y_test).
        """
        # Load dataset
        print("Loading dataset...")
        data = pd.read_csv(self.file_path)

        # Split features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Encode target variable
        print("Encoding target labels...")
        y = self.encoder.fit_transform(y)

        # Scale features
        print("Scaling features...")
        X = self.scaler.fit_transform(X)

        # Train-test split
        print("Splitting dataset into train and test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        print("Dataset processing completed.")
        return X_train, X_test, y_train, y_test


# --- Example Integration with the Main Classes ---
if __name__ == "__main__":
    # Replace 'path_to_dataset.csv' and 'target_column_name' with your actual dataset file path and target column name
    file_path = "path_to_dataset.csv"
    target_column = "anomaly"

    # Initialize dataset handler
    dataset_handler = DatasetHandler(file_path=file_path, target_column=target_column)

    # Load and process dataset
    X_train, X_test, y_train, y_test = dataset_handler.load_and_process_data()

    # Initialize the transportation mode detector
    detector = AnomalyDetector(input_dim=X_train.shape[1], hidden_dim=50, num_layers=3, attention=True)

    # Train the model
    print("Training the model...")
    detector.train(X_train, y_train, epochs=10)

    # Test the model
    print("Testing the model...")
    predictions = detector.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model F1 Score: {f1:.2f}")
