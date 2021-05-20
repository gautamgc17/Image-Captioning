# Automatic Image-Captioning

### Table of Contents:
1. Introduction
2. Applications
3. Prerequisites
4. Data collection
5. Understanding the data
6. Data Cleaning
7. Loading the training set
8. Data Preprocessing — Images
9. Data Preprocessing — Captions
10. Data Preparation using Generator Function
11. Word Embeddings
12. Model Architecture
13. Model Training
14. Making Predictions
15. Conclusion
16. References
### 1. Introduction
Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph. Image captioning, i.e., describing the content observed in an image, has received a significant amount of attention in recent years. It requires both the methods - from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order. Image captioning has many potential applications in real life. A noteworthy one would be to save the captions of an image so that it can be retrieved easily at a later stage just on the basis of this description. It is applicable in various other scenarios, e.g., recommendation in editing applications, usage in virtual assistants, for image indexing, and support of the disabled. With the availability of large datasets, deep neural network (DNN) based methods have been shown to achieve impressive results on image captioning tasks. These techniques are largely based on recurrent neural nets (RNNs), often powered by a Long-Short-Term-Memory (LSTM) component which are quiet useful in sequence data modelling. LSTM nets have been considered as the de-facto standard for vision-language tasks of image captioning , visual question answering , question generation , and visual dialog , due to their compelling ability to memorise long-term dependencies through a memory cell. In this project, CNNs and LSTMs have been used to serve the purpose of the Image Captioning and achieve decent accuracy. 
Figure shown below can be used to understand the task of Image Captioning in a detailed manner.

![image](https://github.com/gautamgc17/Image-Captioning/blob/73487e53c50425c1001f87676e61d8f10b53e135/images/Sample%20IC.jpg)

This figure would be labelled by different people as the following sentences :
- A man and a girl sit on the ground and eat .
- A man and a little girl are sitting on a sidewalk near a blue bag and eating .
- A man wearing a black shirt and a little girl wearing an orange dress share a treat .

But when it comes to machines, automatically generating this textual description from an artificial system is what is called Image Captioning. The task is straightforward – the generated output is expected to describe in a single sentence what is shown in the image – the objects present, their properties, the actions being performed and the interaction between the objects, etc. But to replicate this behaviour in an artificial system is a huge task, as with any other image processing problem and hence the use of complex and advanced techniques such as Deep Learning to solve the task.
### 2. Applications
The main challenge of this task is to capture how objects relate to each other in the image and to express them in a natural language (like English). Some real world scenarios where Image Captioning plays a vital role are as follows :
- Self driving cars — Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self driving system.
- Aid to the blind — We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then the text to voice. Both are now famous applications of Deep Learning.
- CCTV cameras are everywhere today, but along with viewing the world, if we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.
- Automatic Captioning can help, make Google Image Search as good as Google Search, as then every image could be first converted into a caption and then search can be performed based on the caption.
### 3. Prerequisites
Image captioning is an application of one to many type of RNNs. For a given input image model predicts the caption based on the vocabulary of train data using basic Deep Learning techniques. So familarity with concepts like Multi-layered Perceptrons, Convolution Neural Networks, Recurrent Neural Networks, Transfer Learning, Gradient Descent, Backpropagation, Overfitting, Probability, Text Processing, Python syntax and data structures, Keras library etc is necessary. Furthermore, libraries such as cv2, Numpy , keras with Tensorflow backend must be installed.
I have considered the Flickr8k dataset - https://www.kaggle.com/shadabhussain/flickr8k for this project.
### 4. Data Collection
There are many open source datasets available for this problem, like Flickr 8k (It is a collection of 8 thousand described images taken from flickr.com), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But a good dataset to use when getting started with image captioning is the Flickr8K dataset. The reason is because it is realistic and relatively small so that we can download it and build models on our workstation using a CPU(preferably GPU). Flickr8k is a labeled dataset consisting of 8000 photos with 5 captions for each photos. It includes images obtained from the Flickr website.
The images in this dataset are bifurcated as follows:
- Training Set — 6000 images
- Validation Set — 1000 images
- Test Set — 1000 images
### 5. Understanding the Data
In the downloaded Flickr8k dataset, along with Images folder there would be a folder named 'Flickr_TextData' which contains some text files related to the images. One of the files in that folder is “Flickr8k.token.txt” which contains the name of each image along with its 5 captions.

![image](https://github.com/gautamgc17/Image-Captioning/blob/73487e53c50425c1001f87676e61d8f10b53e135/images/Flickr8k.token.txt%20Sample%20.png)

Thus every line contains the <image name>#i <caption>, where 0≤i≤4 , i.e. the name of the image, caption number (0 to 4) and the actual caption.
Firstly, we will create a dictionary named “descriptions” which contains the name of the image (without the .jpg extension) as keys and a list of the 5 captions for the corresponding image as values.

![image](https://github.com/gautamgc17/Image-Captioning/blob/73487e53c50425c1001f87676e61d8f10b53e135/images/descriptions.PNG)
  
Before we proceed with text cleaning, lets visualize an image using Matplotlib library.
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/253e525c7a760f79129845e96d49e2b305a4db40/images/visualization.PNG)
  

### 6. Data Cleaning
When we deal with text, we generally perform some basic cleaning like lower-casing all the words (otherwise “hello” and “Hello” will be regarded as two separate words), remove special tokens or punctuation-marks (like ‘%’, ‘$’, ‘!’, etc.), eliminate words containing numbers (like ‘hey199’, etc.) and in some NLP tasks, we remove stopwords and perform stemming or lemmatization to get root form of the word before finally feeding our textual data to the model. In this project, while text cleaning :

- Stop words have not been removed because if we don’t teach our model how to insert stop words like a, an, the, etc , it would not generate correct english.
- Stemming has not been performed because if we feed in stemmed words, the model is also going to learn those stemmed words . So for example, if the word is ‘running’ and we stem it and make it ‘run’ , the model will predict sentences like “Dog is run” instead of “Dog is running”.
- All the text has been converted to lower case so that ‘the’ and ‘The’ are treated as the same words.
- Numbers, Punctuations and special symbols like ‘@‘, ‘#’ and so on have been removed, so that we generate sentences without any punctuation or symbols. This is beneficial as it helps to reduce the vocabulary size. Small vocabulary size means less number of neurons and hence less parameters to be computed and hence less overfitting.
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/def6374f70f9a537c680bba2b4d6a99e21cab2bd/images/cleaningtext.PNG)

After text cleaning, write all these captions along with their image names in a new file namely, “descriptions.txt” and save it on the disk.
  
#### 6.1 Vocabulary Creation
Next we will create a vocabulary of all the unique words present across all the 8000*5 (i.e. 40000) image captions in the dataset. Total unique words that are there in the dataset are 8424. However, many of these words will occur very few times , say 1, 2 or 3 times. Since it is a predictive sequential model, we would not like to have all the words present in our vocabulary but the words which are more likely to occur or which are common. This helps the model become more robust to outliers and make less mistakes. Hence a threshold has been chosen and if the frequency of the word is less than the threshold frequency (in our case the threshold value chosen is 10), then that particular word is omitted from the vocabulary set. Finally we store the words and their corresponding frequency in a sorted dictionary.
After applying the frequency threshold filter, we get the vocabulary size as 1845 words (having frequency more than 10).
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/3e3f061432e8616b6ec8ccf130a5b7ef42a88340/images/vocab.PNG)
  
Later on, to this vocabulary, we will add two more tokens, namely 'startseq' and 'endseq'. The final vocab size will be total unique words + two extra tokens + 1 (for zero padding). 
  
### 7. Loading the Training Dataset
The dataset also includes “Flickr_8k.trainImages.txt” file which contains the name of the images (or image ids) that belong to the training set. So we need to map these training image ids with the 5 captions corresponding to the image using 'descriptions.txt' file and store the mappings as a dictionary. Another important step while creating the train
  dictionary is to add a ‘startseq’ and ‘endseq’ token in every caption, since RNN or LSTM based layers have been used for generating text. In such layers, the generation of text takes place such that the output of a previous unit acts as an input to the next unit. The model we will develop will generate a caption given a photo, and the caption will be generated one word at a time. The sequence of previously generated words will be provided as input. Therefore, we will need a ‘first word’ to kick-off the generation process and a ‘last word‘ to signal the end of the caption. Hence we need to specify a way which tells the model to stop generating words further. This is accomplished by adding two tokens in the captions i.e.

‘startseq’ -> This is a start sequence token which is added at the start of every caption.
‘endseq’ -> This is an end sequence token which is added at the end of every caption.
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/fd1822c3456dbb0ce20abe699d35515461c2e525/images/train_descriptions.PNG)

### 8. Data Pre-processing : Images
Images are nothing but input X to our model. Any input to a machine or deep learning model must be given in the form of numbers/vectors Hence all the images have to be converted into a fixed size vector which can then be fed as input to a Neural Network. For this purpose, transfer learning has been used.

#### 8.1 Transfer Learning
Transfer learning is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. Transfer learning is popular in deep learning given the enormous resources required to train deep learning models or the large and challenging datasets on which deep learning models are trained. In transfer learning, we can leverage knowledge (features, weights etc) from previously trained models for training newer models and even tackle problems like having less data for the newer task! Thus in this technique, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task.

#### 8.1.1 Pre-Training
When we train the network on a large dataset(for example: ImageNet) , we train all the parameters of the neural network and therefore the model is learned. It may take hours on your GPU. Convnet features are more generic in early layers and more base dataset specific in deeper layers.

#### 8.1.2 Fine Tuning
We can give the new dataset to fine tune the pre-trained CNN. Consider that the new dataset is almost similar to the original dataset used for pre-training. Since the new dataset is similar, the same weights can be used for extracting the features from the new dataset.

When the new dataset is very small, it’s better to train only the final layers of the network to avoid overfitting, keeping all other layers fixed since they contain more generic features and thus we can get weights for our new data. So in this case we remove the final layers of the pre-trained network, add new layers and retrain only the new layers.

When the new dataset is very much large, we can retrain the whole network with initial weights from the pre-trained model.

Remark : How to fine tune if the new dataset is very different from the original dataset ?

Since, the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors), and later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. The earlier layers can help to extract the features of the new data. So it will be good if we fix the earlier layers and retrain the rest of the layers, if we have only small amount of data.

#### 8.2 Image Feature Extraction
In this project, transfer learning has been used to extract features from images. The pre-trained model used is the ResNet model which is a model trained on ImageNet dataset .It has the power of classifying upto 1000 classes. ResNet model has skip connections which means the gradients can flow from one layer to another. This means the gradients can also backpropagate easily and hence ResNet model does not suffer from vanishing gradient problem. Figure shows the architecture of the model.
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/fd1822c3456dbb0ce20abe699d35515461c2e525/images/ResNet%20Architecture.png)

The whole ResNet model has not been trained from scratch. The Convolutional base has been used as a feature extractor. After the convolutional base, a Global average pooling layer has been used to reduce the size of the activation map. Global Average Pooling takes a single channel at a time and averages all the values in that channel to convert it into a single value. The convolutional base produces an activation map of (7,7,2048). The Global Average Pooling layer takes the average of 7*7 (=49) pixels across all the 2048 channels and reduces the size of the activation map to (1,1,2048). So given an image, the model converts it into 2048 dimensional vector. Hence, we just remove the last softmax layer from the model and extract a 2048 length vector (bottleneck features) for every image. These feature vectors are generated for all the images of the training set and later will be sent to the final image captioning model to make predictions. We save all the train image features in a Python dictionary and save it on the disk using Pickle file, namely “encoded_train_img_features.pkl” whose keys are image names and values are corresponding 2048 length feature vector. Similarly we encode all the test images and save their 2048 length vectors on the disk to be used later while making predictions.
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/3666315546042d235a8c316ad078993065e57d4f/images/image_preprocessing.PNG)
