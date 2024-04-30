import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

directory = r'C:\Users\agnix\Downloads\DS691'
classes = [name[:-16] for name in os.listdir(directory) if name.endswith('_input_pairs.npy')]

#print(classes)

reference_text_embeddings = []
image_embeddings_train = []
prompt_embeddings_train = []
image_embeddings_test = []
prompt_embeddings_test = []

for classname in classes:
    file_path = fr'C:\Users\agnix\Downloads\DS691\{classname}_input_pairs.npy'
    data = np.load(file_path)
    num_samples = len(data)
    train_samples = int(0.8 * num_samples)
    
    reference_text_embeddings.append(data[0][0][1])
    for j in data[:train_samples]:
        i = j[0]
        image = i[0]
        text = i[1] 
        image_embeddings_train.append(image)
        prompt_embeddings_train.append(text)

    for j in data[train_samples:]:
        i = j[0]
        image = i[0]
        text = i[1] 
        image_embeddings_test.append(image)
        prompt_embeddings_test.append(text)




# Define the corrected regression-based neural network
class RegressionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Define the KL divergence loss function
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, output, target):
        # Convert output and target to Normal distributions
        output_dist = Normal(output, torch.ones_like(output))
        target_dist = Normal(target, torch.ones_like(target))
        loss = kl_divergence(output_dist, target_dist)
        return loss.mean()

# Example training loop
def train_regression_network(regression_net, optimizer, criterion, image_embeddings, prompt_embeddings, num_epochs=10, batch_size=32):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, len(image_embeddings), batch_size):
            inputs = torch.tensor(image_embeddings[i:i+batch_size], dtype=torch.float32)
            targets = torch.tensor(prompt_embeddings[i:i+batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            outputs = regression_net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            similarities = cosine_similarity(outputs.detach().numpy(), targets.detach().numpy())
            predictions = np.argmax(similarities, axis=1)
            correct += np.sum(predictions == np.arange(len(inputs)))
            total += len(inputs)

        train_accuracy = correct / total * 100
        print('Epoch %d, Loss: %.4f, Train Accuracy: %.2f%%' % (epoch+1, running_loss / len(image_embeddings), train_accuracy))

# Test the regression network using remaining 20% data
def test_regression_network(regression_net, image_embeddings, reference_text_embeddings):
    predicted_prompt_embeddings = []
    correct = 0
    total = 0
    with torch.no_grad():
        for image_embedding in image_embeddings:
            input_image = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
            predicted_embedding = regression_net(input_image)
            predicted_prompt_embeddings.append(predicted_embedding.numpy())

        predicted_prompt_embeddings = np.array(predicted_prompt_embeddings)

        # Compare predicted embeddings with reference embeddings
        for i, predicted_embedding in enumerate(predicted_prompt_embeddings):
            similarities = cosine_similarity(predicted_embedding.reshape(1, -1), reference_text_embeddings)
            max_similarity_index = np.argmax(similarities)
            if max_similarity_index == i % len(classes):  # Assuming classes are ordered
                correct += 1
            total += 1

        test_accuracy = correct / total * 100
        print("Accuracy on test data: %.2f%%" % test_accuracy)

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 768  # Adjusted to match image embeddings size
    output_size = 768  # Adjusted to match text embedding size
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32

    # Initialize regression network, loss function, and optimizer
    regression_net = RegressionNetwork(input_size, output_size)
    criterion = KLDivergenceLoss()
    optimizer = optim.Adam(regression_net.parameters(), lr=learning_rate)

    # Train the regression network
    train_regression_network(regression_net, optimizer, criterion, image_embeddings_train, prompt_embeddings_train, num_epochs, batch_size)

    # Test the regression network
    test_regression_network(regression_net, image_embeddings_test, reference_text_embeddings)