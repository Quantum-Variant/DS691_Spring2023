import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# Define the model architecture
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, contrastive_margin):
        super(MyModel, self).__init__()
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
        self.image_fc = nn.Linear(input_size, hidden_size)
        self.text_fc = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.relu = nn.ReLU()
        self.contrastive_margin = contrastive_margin

    def forward(self, image, text):
        # Tokenize the text
        text_inputs = self.clip_processor(text, return_tensors="pt", padding=True)

        # Preprocess the image and generate the image features
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
        image_embeddings = self.clip_model.get_image_features(**inputs)

        # Generate text features
        text_embeddings = self.clip_model.get_text_features(**text_inputs)

        # Pass image embeddings through fully connected layer
        image_output = self.relu(self.image_fc(image_embeddings))

        # Pass text embeddings through fully connected layer
        text_output = self.relu(self.text_fc(text_embeddings))

        # Concatenate image and text embeddings
        combined_output = torch.cat((image_output, text_output), dim=1)

        # Pass combined embeddings through final fully connected layer
        output = self.fc(combined_output)

        return output, image_embeddings, text_embeddings

# Define data loading and preprocessing
def load_images_and_prompts(data_dir):
    images = []
    prompts = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                prompts.append(f'This is a photo of a {class_name}.')
    return images, prompts

# Define training loop
def train_model(model, images, prompts, criterion_class, criterion_contrastive, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(zip(images, prompts)), total=len(images), desc=f"Epoch {epoch+1}", unit="image")
        for i, (image, prompt) in progress_bar:
            optimizer.zero_grad()
            outputs, image_embeddings, text_embeddings = model(image=image, text=prompt)
            
            # Classification Loss
            class_labels = get_class_labels(prompt, label_map)  # Define your class labels
            loss_class = criterion_class(outputs, torch.tensor([class_labels]))
            
            # Contrastive Loss
            loss_contrastive = compute_contrastive_loss(image_embeddings, text_embeddings, margin=model.contrastive_margin)
            
            # Total Loss
            loss_total = loss_class + loss_contrastive
            
            loss_total.backward()
            optimizer.step()
            running_loss += loss_total.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(images)}")

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([    
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

def generate_class_label_map(dataset_dir):
    class_label_map = {}
    class_index = 0
    for class_name in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, class_name)):
            class_label_map[class_name] = class_index
            class_index += 1
    return class_label_map

def get_class_labels(prompt, label_map):
    # Extract class name from prompt and map it to a numerical label
    class_name = prompt.split()[-1][:-1]
    # Map class name to numerical label (e.g., using a dictionary)
    return label_map[class_name]

# Define contrastive loss function
def compute_contrastive_loss(image_embeddings, text_embeddings, margin):
    # Compute cosine similarity between image and text embeddings
    similarity = torch.cosine_similarity(image_embeddings, text_embeddings, dim=-1)
    # Contrastive loss
    loss = torch.mean(torch.square(torch.max(torch.zeros_like(similarity), margin - similarity)))
    return loss

# Define evaluation function
def evaluate_model(model, images_val, prompts_val, label_map):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, prompt in zip(images_val, prompts_val):
            outputs, image_embeddings, text_embeddings = model(image=image, text=prompt)
            class_name = prompt.split()[-1][:-1]
            class_label = label_map.get(class_name)
            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (predicted == class_label).sum().item()
    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Load images and prompts
data_dir = "data/caltech-101/101_ObjectCategories"
images, prompts = load_images_and_prompts(data_dir)

label_map = generate_class_label_map(data_dir)

# Split data into training and validation sets
images_train, images_val, prompts_train, prompts_val = train_test_split(images, prompts, test_size=0.2, random_state=42)

# Define model hyperparameters
input_size = 512
hidden_size = 256
num_classes = 102  
contrastive_margin = 0.5  # Margin for contrastive loss

# Initialize model, criterion, and optimizer
model = MyModel(input_size, hidden_size, num_classes, contrastive_margin)
criterion_class = nn.CrossEntropyLoss()
criterion_contrastive = nn.MSELoss()  # Example contrastive loss function (you can define your own)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, images_train, prompts_train, criterion_class, criterion_contrastive, optimizer)

# Evaluate the model
evaluate_model(model, images_val, prompts_val, label_map)
