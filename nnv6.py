import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

directory = 'data/caltech-101/101_ObjectCategories'
classes = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

reference_text_embeddings = []
image_embeddings = []
prompt_embeddings = []

for classname in classes:
    file_path = fr'data/caltech-101/Batch files/{classname}_input_pairs.npy'
    data = np.load(file_path)
    num_samples = len(data)
    
    reference_text_embeddings.append(data[0][0][1])
    for j in data:
        i = j[0]
        if len(i) == 2:
            text = i[1]
            i = i[0]
        else:
            text = i
        image_embeddings.append(i)
        prompt_embeddings.append(text)

# Randomize train-test split with 80-20 ratio
image_train, image_test, prompt_train, prompt_test = train_test_split(image_embeddings, prompt_embeddings, test_size=0.8, random_state=42)

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

# Test the regression network using test set
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


# Test the regression network using remaining 20% data
def test_regression_network_knn(regression_net, image_embeddings, reference_embeddings):
    # Ensure the reference embeddings are treated as targets in testing
    reference_embeddings_tensor = torch.tensor(reference_embeddings, dtype=torch.float32)
    
    predicted_prompt_embeddings = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for image_embedding in image_embeddings:
            input_image = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
            predicted_embedding = regression_net(input_image)
            predicted_prompt_embeddings.append(predicted_embedding.squeeze(0))

    predicted_prompt_embeddings = torch.stack(predicted_prompt_embeddings)

    # Initialize Nearest Neighbors model
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(reference_embeddings_tensor)

    # Find the nearest reference embedding for each predicted embedding
    distances, indices = neigh.kneighbors(predicted_prompt_embeddings)

    # Check class alignment using modulo operation based on class counts
    for i, index in enumerate(indices):
        expected_class_index = i % len(classes)
        if classes[expected_class_index] == classes[index[0] % len(classes)]:
            correct += 1
        total += 1

    test_accuracy = correct / total * 100
    print("Accuracy on test data using k-NN: %.2f%%" % test_accuracy)

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 768  # Adjusted to match image embeddings size
    output_size = 768  # Adjusted to match text embedding size
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 8

    # Initialize regression network, loss function, and optimizer
    regression_net = RegressionNetwork(input_size, output_size)
    criterion = KLDivergenceLoss()
    optimizer = optim.Adam(regression_net.parameters(), lr=learning_rate)

    # Train the regression network
    train_regression_network(regression_net, optimizer, criterion, image_train, prompt_train, num_epochs, batch_size)

    # Test the regression network
    test_regression_network(regression_net, image_test, reference_text_embeddings)
