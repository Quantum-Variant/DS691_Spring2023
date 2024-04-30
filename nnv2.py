import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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
        image = i[0]
        text = i[1] 
        image_embeddings.append(image)
        prompt_embeddings.append(text)

# Randomize train-test split with 80-20 ratio
image_train, image_test, prompt_train, prompt_test = train_test_split(image_embeddings, prompt_embeddings, test_size=0.2, random_state=42)

# Define the siamese network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, x):
        return self.fc(x)

# Define the loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Example training loop
def train_siamese_network(siamese_net, optimizer, criterion, image_embeddings, prompt_embeddings, num_epochs=10, batch_size=32):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, len(image_embeddings), batch_size):
            inputs = torch.tensor(image_embeddings[i:i+batch_size], dtype=torch.float32)
            prompts = torch.tensor(prompt_embeddings[i:i+batch_size], dtype=torch.float32)
            labels = torch.ones(len(inputs))
            optimizer.zero_grad()
            outputs1 = siamese_net(inputs)
            outputs2 = prompts
            loss = criterion(outputs1, outputs2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            similarities = cosine_similarity(outputs1.detach().numpy(), prompts.detach().numpy())
            predictions = np.argmax(similarities, axis=1)
            correct += np.sum(predictions == np.arange(len(inputs)))
            total += len(inputs)

        train_accuracy = correct / total * 100
        print('Epoch %d, Loss: %.4f, Train Accuracy: %.2f%%' % (epoch+1, running_loss / len(image_embeddings), train_accuracy))

# Test the siamese network using remaining 20% data
def test_siamese_network(siamese_net, image_embeddings, reference_text_embeddings):
    predicted_prompt_embeddings = []
    correct = 0
    total = 0
    with torch.no_grad():
        for image_embedding in image_embeddings:
            input_image = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
            predicted_embedding = siamese_net(input_image)
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
    embedding_size = 768  # Adjusted to match image embeddings size
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 16

    # Initialize siamese network, loss function, and optimizer
    siamese_net = SiameseNetwork(embedding_size)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)

    # Train the siamese network
    train_siamese_network(siamese_net, optimizer, criterion, image_train, prompt_train, num_epochs, batch_size)

    # Test the siamese network
    test_siamese_network(siamese_net, image_test, reference_text_embeddings)
