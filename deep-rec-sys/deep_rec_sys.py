# Deep Recommender Systems
# This script demonstrates the implementation of deep recommender systems using PyTorch.
# It includes various models: NCF, Wide & Deep, and NeuMF, and evaluates their performance.
# The script also includes synthetic data generation with non-linear patterns and visualization of model performance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class RatingDataset(Dataset):
    """Dataset for rating prediction"""
    
    def __init__(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        self.ratings_df = ratings_df
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        # Convert to indices
        self.user_indices = torch.tensor([
            self.user_mapping[user] for user in ratings_df[user_col]
        ], dtype=torch.long)
        
        self.item_indices = torch.tensor([
            self.item_mapping[item] for item in ratings_df[item_col]
        ], dtype=torch.long)
        
        self.ratings = torch.tensor(ratings_df[rating_col].values, dtype=torch.float)
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_idx': self.user_indices[idx],
            'item_idx': self.item_indices[idx],
            'rating': self.ratings[idx]
        }

class NCF(nn.Module):
    """Neural Collaborative Filtering"""
    
    def __init__(self, n_users, n_items, n_factors=10, layers=[20, 10], dropout=0.1):
        super(NCF, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.mlp_layers = []
        input_size = 2 * n_factors
        
        for layer_size in layers:
            self.mlp_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # Get embeddings
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        
        # Concatenate
        concat = torch.cat([user_embed, item_embed], dim=1)
        
        # MLP
        mlp_output = self.mlp(concat)
        
        # Output
        output = self.output_layer(mlp_output)
        
        return output.squeeze()

class WideAndDeep(nn.Module):
    """Wide & Deep Learning Model"""
    
    def __init__(self, n_users, n_items, n_factors=10, deep_layers=[20, 10], dropout=0.1):
        super(WideAndDeep, self).__init__()
        
        # Wide component (linear)
        self.wide_user_embedding = nn.Embedding(n_users, 1)
        self.wide_item_embedding = nn.Embedding(n_items, 1)
        
        # Deep component
        self.deep_user_embedding = nn.Embedding(n_users, n_factors)
        self.deep_item_embedding = nn.Embedding(n_items, n_factors)
        
        # Deep MLP
        self.deep_layers = []
        input_size = 2 * n_factors
        
        for layer_size in deep_layers:
            self.deep_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.deep_mlp = nn.Sequential(*self.deep_layers)
        
        # Output layer
        self.output_layer = nn.Linear(deep_layers[-1] + 2, 1)  # +2 for wide features
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # Wide component
        wide_user = self.wide_user_embedding(user_idx).squeeze()
        wide_item = self.wide_item_embedding(item_idx).squeeze()
        wide_features = torch.stack([wide_user, wide_item], dim=1)
        
        # Deep component
        deep_user = self.deep_user_embedding(user_idx)
        deep_item = self.deep_item_embedding(item_idx)
        deep_concat = torch.cat([deep_user, deep_item], dim=1)
        deep_output = self.deep_mlp(deep_concat)
        
        # Combine wide and deep
        combined = torch.cat([wide_features, deep_output], dim=1)
        output = self.output_layer(combined)
        
        return output.squeeze()

class NeuMF(nn.Module):
    """Neural Matrix Factorization"""
    
    def __init__(self, n_users, n_items, n_factors=10, mlp_layers=[20, 10], dropout=0.1):
        super(NeuMF, self).__init__()
        
        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, n_factors)
        self.gmf_item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(n_users, n_factors)
        self.mlp_item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        self.mlp_layers = []
        input_size = 2 * n_factors
        
        for layer_size in mlp_layers:
            self.mlp_layers.extend([
                nn.Linear(input_size, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = layer_size
        
        self.mlp = nn.Sequential(*self.mlp_layers)
        
        # Output layer
        self.output_layer = nn.Linear(n_factors + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_idx, item_idx):
        # GMF component
        gmf_user = self.gmf_user_embedding(user_idx)
        gmf_item = self.gmf_item_embedding(item_idx)
        gmf_output = gmf_user * gmf_item  # Element-wise product
        
        # MLP component
        mlp_user = self.mlp_user_embedding(user_idx)
        mlp_item = self.mlp_item_embedding(item_idx)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_concat)
        
        # Combine
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output_layer(combined)
        
        return output.squeeze()

def train_model(model, train_loader, val_loader, n_epochs=50, learning_rate=0.001, device='cpu'):
    """Train a deep recommendation model"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(user_idx, item_idx)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                user_idx = batch['user_idx'].to(device)
                item_idx = batch['item_idx'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(user_idx, item_idx)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)
            
            preds = model(user_idx, item_idx)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return mae, rmse, predictions, actuals

# Generate synthetic data
np.random.seed(42)
n_users = 500
n_items = 300
n_ratings = 3000

# Create synthetic ratings with non-linear patterns
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(5, 20)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create non-linear patterns
        user_factor = np.random.normal(0, 1)
        item_factor = np.random.normal(0, 1)
        
        # Non-linear interaction
        interaction = np.sin(user_factor) * np.cos(item_factor) + user_factor * item_factor
        
        # Add noise and convert to rating
        rating = max(1, min(5, 3 + interaction + np.random.normal(0, 0.3)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with Non-linear Patterns:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")

# Prepare data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = RatingDataset(train_df)
val_dataset = RatingDataset(val_df)
test_dataset = RatingDataset(test_df)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train different models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

models = {
    'NCF': NCF(train_dataset.n_users, train_dataset.n_items, n_factors=10, layers=[20, 10]),
    'Wide&Deep': WideAndDeep(train_dataset.n_users, train_dataset.n_items, n_factors=10, deep_layers=[20, 10]),
    'NeuMF': NeuMF(train_dataset.n_users, train_dataset.n_items, n_factors=10, mlp_layers=[20, 10])
}

results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs=50, device=device)
    
    # Evaluate
    mae, rmse, predictions, actuals = evaluate_model(model, test_loader, device=device)
    
    results[name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'actuals': actuals
    }
    
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Training curves
plt.subplot(3, 4, 1)
for name, result in results.items():
    plt.plot(result['train_losses'], label=f'{name} Train')
    plt.plot(result['val_losses'], label=f'{name} Val', linestyle='--')
plt.title('Training Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Model comparison - MAE
plt.subplot(3, 4, 2)
mae_values = [results[name]['mae'] for name in results.keys()]
plt.bar(results.keys(), mae_values, color=['blue', 'red', 'green'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')

# Plot 3: Model comparison - RMSE
plt.subplot(3, 4, 3)
rmse_values = [results[name]['rmse'] for name in results.keys()]
plt.bar(results.keys(), rmse_values, color=['blue', 'red', 'green'])
plt.title('RMSE Comparison')
plt.ylabel('Root Mean Square Error')

# Plot 4-6: Prediction vs Actual for each model
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 4, 4 + i)
    plt.scatter(result['actuals'], result['predictions'], alpha=0.6)
    plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')

# Plot 7: Rating distribution
plt.subplot(3, 4, 7)
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Plot 8: Model architecture comparison
plt.subplot(3, 4, 8)
architectures = ['NCF', 'Wide&Deep', 'NeuMF']
parameters = [
    sum(p.numel() for p in models[name].parameters()) 
    for name in architectures
]
plt.bar(architectures, parameters)
plt.title('Model Parameters')
plt.ylabel('Number of Parameters')

# Plot 9: Training time comparison (simulated)
plt.subplot(3, 4, 9)
training_times = [50, 45, 55]  # Simulated times
plt.bar(architectures, training_times)
plt.title('Training Time (epochs)')
plt.ylabel('Time')

# Plot 10: Convergence comparison
plt.subplot(3, 4, 10)
for name, result in results.items():
    plt.plot(result['val_losses'], label=name, marker='o', markersize=3)
plt.title('Convergence Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()

# Plot 11: Error distribution
plt.subplot(3, 4, 11)
for name, result in results.items():
    errors = np.array(result['predictions']) - np.array(result['actuals'])
    plt.hist(errors, bins=20, alpha=0.7, label=name)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()

# Plot 12: Model summary
plt.subplot(3, 4, 12)
summary_data = {
    'Model': list(results.keys()),
    'MAE': [results[name]['mae'] for name in results.keys()],
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'Parameters': parameters
}
summary_df = pd.DataFrame(summary_data)
plt.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
plt.axis('off')
plt.title('Model Summary')

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Analysis ===")

# Compare model performance
print("Model Performance Comparison:")
for name, result in results.items():
    print(f"{name}:")
    print(f"  MAE: {result['mae']:.4f}")
    print(f"  RMSE: {result['rmse']:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in result['model'].parameters()):,}")
    print()

# Analyze prediction patterns
print("Prediction Pattern Analysis:")
for name, result in results.items():
    predictions = np.array(result['predictions'])
    actuals = np.array(result['actuals'])
    
    print(f"{name}:")
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Prediction std: {predictions.std():.3f}")
    print(f"  Bias: {predictions.mean() - actuals.mean():.3f}")
    print()

# Test recommendations
print("Recommendation Test:")
test_user = 0
test_item = 0

# Find user and item in test set
user_mapping = {user: idx for idx, user in enumerate(ratings_df['user_id'].unique())}
item_mapping = {item: idx for idx, item in enumerate(ratings_df['item_id'].unique())}

if test_user in user_mapping and test_item in item_mapping:
    user_idx = torch.tensor([user_mapping[test_user]]).to(device)
    item_idx = torch.tensor([item_mapping[test_item]]).to(device)
    
    print(f"Predictions for User {test_user}, Item {test_item}:")
    for name, result in results.items():
        model = result['model'].to(device)
        model.eval()
        with torch.no_grad():
            pred = model(user_idx, item_idx).cpu().numpy()[0]
        print(f"  {name}: {pred:.3f}")