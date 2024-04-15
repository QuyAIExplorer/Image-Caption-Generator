### Image Captioning with Transfer Learning and LSTM
This project aims to automatically generate captions for images by extracting features from them using a pretrained model (transfer learning) and matching these features with corresponding captions. The project is implemented using PyTorch and focuses on understanding and implementing neural networks, transfer learning, and LSTM models for natural language processing tasks.

### Approach
-Dataset Preparation: Prepare a dataset of images with corresponding captions. The dataset should be suitable for training an image captioning model.

-Model Architecture: Use a pretrained CNN model (Inception v3) to extract features from images. These features are then fed into an LSTM network to generate captions.

-Training: Train the model using the prepared dataset. The model learns to generate captions based on the features extracted from images.

-Evaluation: Evaluate the model's performance using metrics such as BLEU score and human evaluation to assess the quality of generated captions.

-Deployment: Deploy the trained model to generate captions for new images automatically.

### Results
The project has shown promising results in generating captions for images. Although the generated captions may not match the original captions word for word, the model is capable of providing more detailed descriptions of different parts of the image compared to the original caption. This can be observed in the result shown in Result.png, where the model describes various elements of the image in detail.

### Requirements
Download the dataset used: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
Then set images folder, captions.txt in the same folder with main code.

train.py: For training the network

model.py: creating the encoderCNN, decoderRNN and hooking them togethor

get_loader.py: Loading the data, creating vocabulary

utils.py: Load model, save model, printing few test cases downloaded online

### Installation
1. Clone the repository:
git clone https://github.com/QuyAIExplorer/Image-Caption-Generator.git

### Usage
1. Download the English language model for spaCy by running:
python -m spacy download en_core_web_sm
2. Modify the paths in the train_loader function in train.py to point to your dataset.
3. Train and test the model by running: train.py

