import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_rows, img_cols = 200, 200
img_channels = 1
batch_size = 32
nb_classes = 5
nb_epoch = 15
nb_filters = 32
nb_pool = 2
nb_conv = 3

output = ["OK", "NOTHING", "PEACE", "PUNCH", "STOP"]

current_gesture = "Waiting..."

class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
  
        self.conv1 = nn.Conv2d(img_channels, nb_filters, kernel_size=nb_conv, padding='valid')
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=nb_conv)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(nb_pool, nb_pool)
        self.dropout1 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self._to_linear = self._get_conv_output((1, img_channels, img_rows, img_cols))
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, nb_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape[1:]))
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool(output)
        output = self.dropout1(output)
        n_size = output.data.view(batch_size, -1).size(1)
        return n_size
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # softmax在損失函數中

class GestureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data():
    path2 = 'D:/chouai/CV_final/CNNGestureRecognizer-master/imgfolder_b'
    images = []
    labels = []
    
    # 創建手勢類型到標籤的映射
    gesture_prefixes = {
        'iiiok': 0,     # OK
        'iiok': 0,      # OK
        'iok': 0,       # OK
        'nnnothing': 1, # NOTHING
        'nnothing': 1,  # NOTHING
        'nothing': 1,   # NOTHING
        'pppeace': 2,   # PEACE
        'ppeace': 2,    # PEACE
        'peace': 2,     # PEACE
        'pppunch': 3,   # PUNCH
        'ppunch': 3,    # PUNCH
        'punch': 3,     # PUNCH
        'ssstop': 4,    # STOP
        'sstop': 4,     # STOP
        'stop': 4       # STOP
    }
    
    print("Loading images from:", path2)
    
    for img_file in os.listdir(path2):
        try:
            prefix = next((key for key in gesture_prefixes.keys() if img_file.startswith(key)), None)
            if prefix is None:
                continue
                
            img_path = os.path.join(path2, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_rows, img_cols))
                images.append(img)
                labels.append(gesture_prefixes[prefix])
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print(f"Loaded {len(images)} images")
    
    if len(images) == 0:
        raise ValueError("No images were loaded. Check the data directory and file names.")
    
    images = np.array(images)
    labels = np.array(labels)
    
    images = images.astype('float32')
    images /= 255.0
    
    for i in range(nb_classes):
        count = np.sum(labels == i)
        print(f"Class {output[i]}: {count} samples")
    
    return images, labels

def train_model(model_path='best_model.pth'):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    print("Loading training data...")
    images, labels = load_data()
    
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    train_dataset = GestureDataset(X_train, y_train, transform)
    val_dataset = GestureDataset(X_val, y_val, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = GestureCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    print("Starting training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(nb_epoch):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if epoch == 0 or val_loss < min(history['val_loss'][:-1]):
            print(f'Saving model with val_loss: {val_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def guess_gesture(model, img):
    """預測手勢"""
    global current_gesture
    try:
        img_pil = Image.fromarray(img)
        
        transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        img_tensor = transform(img_pil).unsqueeze(0)

        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # 預測
        with torch.no_grad():
            model.eval()  # 確保模型在評估模式
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1).item()
            
            # 獲取所有類別的概率
            probs = probabilities[0].cpu().numpy() * 100
            
            prediction_dict = {}
            for i, prob in enumerate(probs):
                if i < len(output): 
                    prediction_dict[str(output[i])] = float(prob)

            try:
                with open('gesture_prob.json', 'w') as f:
                    json.dump(prediction_dict, f)
            except Exception as e:
                print(f"Warning: Could not write to gesture_prob.json: {str(e)}")
            
            # 更新當前手勢
            if predicted < len(output):  # 確保預測索引在有效範圍內
                current_gesture = output[predicted]
            else:
                current_gesture = "NOTHING"
                predicted = 1  # 設置為 NOTHING 的索引
            
            return predicted
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        current_gesture = "NOTHING"
        return 1  # 返回 NOTHING 作為默認值

def preprocess_image(img):
    """預處理圖像"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_model():
    try:
        device = torch.device("cpu") 
        model = GestureCNN().to(device)

        checkpoint = torch.load(
            'D:/chouai/CV_final/CNNGestureRecognizer-master/best_model.pth',
            map_location=device
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() 
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    train_model()
