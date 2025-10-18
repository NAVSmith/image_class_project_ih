# Deep Learning Image Classification Project

This project implements a CNN-based image classification system using either CIFAR-10 or Animals10 datasets. The project includes both custom CNN architectures and transfer learning approaches, with a complete Flask web application for deployment.

## 🎯 Project Objectives

- Build and compare custom CNN architectures with transfer learning models
- Achieve >70% validation accuracy for passing grade
- Deploy the best model via Flask web application
- Provide comprehensive analysis and documentation

## 📁 Project Structure

```
image_class_project_ih/
├── .github/
│   └── copilot-instructions.md    # AI coding assistant guidelines
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb  # Dataset analysis and visualization
│   ├── 02_custom_cnn.ipynb       # Custom CNN implementation
│   └── 03_transfer_learning.ipynb # Transfer learning experiments
├── src/                           # Python source modules
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── model_architectures.py     # CNN and transfer learning models
│   ├── training_utils.py         # Training and evaluation utilities
│   └── train_model.py            # Command-line training script
├── app/                          # Flask web application
│   ├── app.py                    # Main Flask application
│   ├── utils.py                  # Model loading and prediction utilities
│   ├── templates/                # HTML templates
│   │   ├── upload.html           # Image upload interface
│   │   └── results.html          # Results display
│   └── static/                   # CSS and static files
│       └── style.css             # Custom styling
├── models/                       # Saved trained models
├── data/                        # Dataset storage (gitignored)
├── reports/                     # PDF reports and visualizations
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd image_class_project_ih

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Exploration

Start with the data exploration notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Model Training

Train models using the notebooks or command-line script:

```bash
# Using command-line script
python src/train_model.py --model-type custom --dataset cifar10 --epochs 50
python src/train_model.py --model-type transfer --base-model vgg16 --dataset cifar10 --epochs 30
```

### 4. Web Application Deployment

Deploy the trained model via Flask:

```bash
cd app
python app.py --model ../models/best_model.h5 --dataset cifar10
```

Access the application at `http://localhost:5000`

## 📊 Datasets

### CIFAR-10

- **Images**: 60,000 (50,000 train + 10,000 test)
- **Size**: 32×32 pixels, RGB
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Download**: Automatically loaded via `tf.keras.datasets.cifar10`

### Animals10 (Alternative)

- **Images**: ~28,000
- **Classes**: 10 (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant)
- **Download**: [Kaggle Animals10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)

## 🔧 Model Architectures

### Custom CNN

- **Architecture**: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
- **Blocks**: 3 convolutional blocks with increasing filters (32, 64, 128)
- **Head**: GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(classes)
- **Parameters**: ~500K

### Transfer Learning

- **Base Models**: VGG16, ResNet50, EfficientNetB0
- **Strategy**: Frozen base + custom classification head
- **Fine-tuning**: Unfreeze base layers with lower learning rate
- **Input Size**: 224×224 (resized from original)

## 📈 Training Configuration

### Data Preprocessing

- **Normalization**: Pixel values scaled to [0, 1]
- **Augmentation**: Rotation (±15°), shifts (±10%), horizontal flip, zoom (±10%)
- **Splits**: 70% train / 15% validation / 15% test

### Training Parameters

- **Optimizer**: Adam (lr=0.001) or SGD with momentum
- **Loss**: Categorical crossentropy
- **Batch Size**: 32
- **Epochs**: 50-100 (custom CNN), 20-50 (transfer learning)
- **Callbacks**: Early stopping, model checkpoint, learning rate reduction

## 🎯 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Detailed classification analysis
- **Training curves**: Loss and accuracy over epochs

## 🌐 Web Application Features

### Upload Interface

- **Multi-file upload**: Drag & drop or browse
- **Preview**: Image thumbnails before processing
- **Validation**: File type and size checking

### Results Display

- **Predictions**: Top class with confidence score
- **Probabilities**: All class probabilities visualization
- **Export**: JSON/CSV results download
- **API**: RESTful endpoint for programmatic access

### API Usage

```bash
# Single prediction
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict

# Response
{
  "prediction": "cat",
  "confidence": 0.89,
  "all_probabilities": {
    "cat": 0.89,
    "dog": 0.07,
    "bird": 0.02,
    ...
  }
}
```

## 📋 Development Workflow

### 1. Data Analysis

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Custom CNN Development

```bash
jupyter notebook notebooks/02_custom_cnn.ipynb
```

### 3. Transfer Learning Experiments

```bash
jupyter notebook notebooks/03_transfer_learning.ipynb
```

### 4. Model Comparison

- Compare custom CNN vs transfer learning
- Select best performing model
- Document performance differences

### 5. Deployment

```bash
cd app
python app.py --model ../models/best_model.h5
```

## 🏆 Success Criteria

- ✅ **Accuracy**: >70% validation accuracy (passing grade)
- ✅ **Architecture**: Both custom CNN and transfer learning implemented
- ✅ **Comparison**: Detailed performance analysis
- ✅ **Deployment**: Working Flask application
- ✅ **Documentation**: Comprehensive report and code documentation

## 🔍 Model Performance Tracking

Track key metrics throughout development:

```python
# Example performance tracking
{
    "custom_cnn": {
        "validation_accuracy": 0.75,
        "test_accuracy": 0.73,
        "parameters": 512000,
        "training_time": "45 minutes"
    },
    "vgg16_transfer": {
        "validation_accuracy": 0.82,
        "test_accuracy": 0.80,
        "parameters": 15000000,
        "training_time": "25 minutes"
    }
}
```

## 🚀 Advanced Features

### Model Serving

- **TensorFlow Serving**: Production deployment option
- **Docker**: Containerized deployment
- **Cloud deployment**: AWS/GCP integration

### Performance Optimization

- **Model quantization**: Reduce model size
- **Batch prediction**: Efficient multi-image processing
- **Caching**: Model and prediction caching

## 📖 Additional Resources

- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Flask Deployment Best Practices](https://flask.palletsprojects.com/en/2.0.x/deploying/)

## 🤝 Contributing

1. Follow the coding patterns in `.github/copilot-instructions.md`
2. Use the established project structure
3. Add tests for new functionality
4. Update documentation

## 📄 License

This project is for educational purposes as part of the Ironhack Data Science program.
