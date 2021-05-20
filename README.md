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
Image captioning is an application of one to many type of RNNs. For a given input image model predicts the caption based on the vocabulary of train data using basic Deep Learning techniques. So familarity with concepts like Multi-layered Perceptrons, Convolution Neural Networks, Recurrent Neural Networks, Transfer Learning, Gradient Descent, Backpropagation, Overfitting, Probability, Text Processing, Python syntax and data structures, Keras library etc is necessary. 
I have considered the Flickr8k dataset - https://www.kaggle.com/shadabhussain/flickr8k for this project.
### 4. Data Collection
There are many open source datasets available for this problem, like Flickr 8k (containing8k images), Flickr 30k (containing 30k images), MS COCO (containing 180k images), etc. But a good dataset to use when getting started with image captioning is the Flickr8K dataset. The reason is because it is realistic and relatively small so that we can download it and build models on our workstation using a CPU(preferably GPU). Flickr8k is a labeled dataset consisting of 8000 photos with 5 captions for each photos. It includes images obtained from the Flickr website.
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
  
#### 5.1 Vocabulary Creation
Next we will create a vocabulary of all the unique words present across all the 8000*5 (i.e. 40000) image captions in the dataset. Total unique words that are there in the dataset are 8424. However, many of these words will occur very few times , say 1, 2 or 3 times. Since it is a predictive sequential model, we would not like to have all the words present in our vocabulary but the words which are more likely to occur or which are common. This helps the model become more robust to outliers and make less mistakes. Hence a threshold has been chosen and if the frequency of the word is less than the threshold frequency (in our case the threshold value chosen is 10), then that particular word is omitted from the vocabulary set. Finally we store the words and their corresponding frequency in a sorted dictionary.
After applying the frequency threshold filter, we get the vocabulary size as 1845 words (having frequency more than 10).
  
![image](https://github.com/gautamgc17/Image-Captioning/blob/3e3f061432e8616b6ec8ccf130a5b7ef42a88340/images/vocab.PNG)
  
Later on, to this vocabulary, we will add two more tokens, namely 'startseq' and 'endseq'. The final vocab size will be total unique words + two extra tokens + 1 (for zero padding). 
  



#### 3.2 Loading the Training Set
The dataset also includes “Flickr_8k.trainImages.txt” file which contains the name of the images (or image ids) that belong to the training set. So all the training image ids have been mapped with the captions and stored in a dictionary. Another important step while creating the dictionary was to add a ‘startseq’ and ‘endseq’ token in every caption, since RNN or LSTM based layers have been used for generating text. In such layers, the generation of text takes place such that the output of a previous unit acts as an input to the next unit. So such model can keep generating words for infinite time steps. Hence we need to specify a way which tells the model to stop generating words further. This is accomplished by adding two tokens in the captions i.e.

‘startseq’ -> This is a start sequence token which was added at the start of every caption.
‘endseq’ -> This is an end sequence token which was added at the end of every caption.
The model will be able to generate this token only if we have it in our training set.

4. Data Pre-processing : Images
Images are nothing but input (X) to our model. Any input to a model must be given in the form of a vector. Hence all the images have to be converted into a fixed size vector which can then be fed as input to a Neural Network. For this purpose, transfer learning has been used.

4.1 Transfer Learning
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem [8]. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. Transfer learning is popular in deep learning given the enormous resources required to train deep learning models or the large and challenging datasets on which deep learning models are trained. In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task.

4.1.1 Pre-Training
When we train the network on a large dataset(for example: ImageNet) , we train all the parameters of the neural network and therefore the model is learned. It may take hours on your GPU.

4.1.2 Fine Tuning
We can give the new dataset to fine tune the pre-trained CNN. Consider that the new dataset is almost similar to the original dataset used for pre-training. Since the new dataset is similar, the same weights can be used for extracting the features from the new dataset.

When the new dataset is very small, it’s better to train only the final layers of the network to avoid overfitting, keeping all other layers fixed. So in this case we remove the final layers of the pre-trained network, add new layers and retrain only the new layers.

When the new dataset is very much large, we can retrain the whole network with initial weights from the pre-trained model.

Remark : How to fine tune if the new dataset is very different from the original dataset ?

The earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors), but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. The earlier layers can help to extract the features of the new data. So it will be good if we fix the earlier layers and retrain the rest of the layers, if we have only small amount of data.

4.2 Image Feature Extraction
In this project, transfer learning has been used to extract features from images. The pre-trained model used is the ResNet model which is a model trained on ImageNet dataset [9]. It has the power of classifying upto 1000 classes. ResNet model has skip connections which means the gradients can flow from one layer to another. This means the gradients can also backpropagate easily and hence ResNet model does not suffer from vanishing gradient problem. Figure 2 shows the architecture of the ResNet model.



The whole ResNet model has not been trained from scratch. The Convolutional base has been used as a feature extractor. After the convolutional base, a Global average pooling layer has been used to reduce the size of the activation map. Global Average Pooling takes a single channel at a time and averages all the values in that channel to convert it into a single value. The convolutional base produces an activation map of (7,7,2048). The Global Average Pooling layer takes the average of 7*7 (=49) pixels across all the 2048 channels and reduces the size of the activation map to (1,1,2048). So given an image, the model converts it into 2048 dimensional vector. These feature vectors are generated for all the images of the training set and later will be sent to the final image captioning model to make predictions. Similarly we encode all the test images and save their 2048 length vectors on the disk to be used later.

5. Data Pre-processing : Captions
Captions are something that will be predicted by the model. So during the training period, captions are the target variables (Y) that the model is learning to predict. But the prediction of the entire caption, given the image does not happen at once. Caption has to be predicted word by word. Thus each word has to be encoded into a fixed size vector. To map each word in our vocabulary to some index, a python dictionary called word_to_idx has been created. Also the model outputs numbers which have to be decoded to form captions. Hence another python dictionary called idx_to_word to map each index with a word in the vocabulary has been created. These two Python dictionaries have been used as follows:

word_to_idx[‘abc’] -> returns index of the word ‘abc’
idx_to_word[k] -> returns the word whose index is ‘k’
When the model is given a batch of sentences as input, the sentences maybe of different lengths. Hence to complete the 2D matrix or batch of sentences, zeros have been filled in for shorter sentences to make them equal in length to the longer sentences. The length of all the sentences have been fixed i.e equal to the length of the longest sentence in our vocabulary.

6. Data Preparation using Generator Function
This is one of the most important steps in this project. In this step the data has been formulated in a manner which will be convenient to be given as input to the deep learning model. This step is explained by taking an example [10] as follows :

6.1 Understanding Generator function with the help of an example.
Let us consider we have 3 images and their 3 corresponding captions as shown in Figure 3,4 and 5.



The first two images and their captions are used to train the model and the third image is used to test the model. The challenge is to frame this as a supervised learning problem, how does the data matrix look like and how many data points we have. Firstly, the images need to be converted into their corresponding 2048 length feature vector. Let Image_1 and Image_2 be the feature vectors of of train images viz Figure 3 and Figure 4. Secondly, a vocabulary is built for the first two(train) captions by adding the two tokens “startseq” and “endseq” in both the captions.

Caption_1 -> “startseq the black cat sat on grass endseq”=
Caption_2 -> “startseq the white cat is walking on road endseq”
-> vocab = {black, cat, endseq, grass, is, on, road, sat, startseq, the, walking, white}

An index is given to each word in the vocabulary as follows :

black-1, cat-2, endseq-3, grass-4, is-5, on-6, road-7, sat-8, startseq-9, the-10, walking-11, white-12

Now this can be framed as a supervised learning problem where we have a set of data points D = {Xi,Yi}, where Xi is the feature vector of data point ‘i’ and Yi is the corresponding target variable. Image vector is the input and the caption is what we need to predict. But the way in which a caption is predicted is as follows :

In the first step we provide the image vector and the first word as input and try to predict the second word i.e. Input = Image_1 + ‘startseq’; Output = ‘the’
Then we provide image vector and the first two words as input and try to predict the third word, i.e. Input = Image_1 + ‘startseq the’; Output = ‘cat’
And so on…
Thus the data points for one image and its corresponding caption can be summarised as shown in figure 6.



NOTE : One image+caption is not a single data point but are multiple data points depending on the length of the caption.

Similarly if we consider both the images and their captions, the data matrix will look as shown in figure 7.



Thus, we can conclude from the example that in every data point, it’s not just the image that goes as input to the system, but also, a partial caption which helps to predict the next word in the sequence. However we cannot pass the actual English text of the caption, rather we pass the sequence of indices where each index represents a unique word. Since we had already created a dictionary word_to_idx, the data matrix after replacing the words with their indices is shown in figure 8.



The model uses batch processing and due to that we need to make sure that each sequence is of equal length. Hence we need to append 0’s (zero padding) at the end of each sequence. For this we find out the maximum length, a caption has in the whole dataset. The maximum length of a caption in our dataset is 34. So we append those many number of zeros which will lead to every sequence having a length of 34. The data matrix will then look as shown in figure 9.



In this example we had considered only 2 images and captions which lead to a generation of 15 data points. However in our actual training dataset there are 6000 images, each having 5 captions. This makes a total of 30,000 images and captions. Even if we assume that each caption is just 7 words long, it will lead to a total of 30000*7 = 210000 data points.

Size of the data matrix = n*m , where  n-> number of data points (assumed as 210000) and m-> length of each data point

Clearly m = Length of image vector(2048) + Length of partial caption(x) = 2048 + x

x here is not equal to 34. This is because every word will be mapped to a higher dimensional space through some word embedding techniques. In this project , instead of training an embedded layer from scratch, Glove vectors have been used which is again an application of transfer learning. Glove vectors convert each word into a 200-dimensional vector. Since each partial caption contains 34 indices , where each index is a vector of length 200. Therefore, x = 34*200 = 6800. Hence m = 2048 + 6800 = 8848. Finally the size of the data matrix = 210000 * 8848 = 1,85,80,80,000 blocks. Now even if we assume that 1 block takes 2 byte, then, to store this data matrix, we will require more than 3GB of main memory. This is a very huge requirement and will make the system very slow. For this reason I have used Data Generator which is a functionality that is natively implemented in python. With SGD, we do not calculate the loss on the entire data set to update the gradients. Rather in every iteration, we calculate the loss on a batch of data points (typically 64, 128, 256, etc.) to update the gradients [9]. This means that we do not require to store the entire dataset in the memory at once. Even if we have the current batch of points in the memory, it is sufficient for our purpose. A generator function in Python is used exactly for this purpose. It’s like an iterator which resumes the functionality from the point it left the last time it was called.

6.2 Word Embeddings - Transfer Learning
This section describes how indices of words of a caption have been converted into embeddings of fixed length. Whenever we feed data into RNN or LSTM layer, this data should have also been passed through the embedding layer. This embedding layer can be trained or we can pre - initialise this layer. In this project we have pre - initialised this layer by using Glove vectors from the file Glove6B200D.txt . This txt file contains 200 dimensional word embeddings for 6 billion words [11]. All 6 billion words are not needed and we just need the embeddings for the words that are there in our vocabulary.

Note : Words that are present in our vocab but are not there in the glove embeddings file will be substituted by all zeros(200-dimensional).

7. Image Captioning - Model Architecture
Image feature vector along with the partial sequence(caption) will be given to the model and the next word in the sequence is generated as the output. Then the output is again appended to the input and next word in the sequence is generated . This process continues until the model generates an ‘endseq’ token which marks the end of the caption. Figure 10 shows the high level overview of the model architecture.



Since the input consists of two parts, an image vector and a partial caption, the Sequential API provided by the Keras library cannot be used. For this reason, the Functional API has been used which allows to create Merge Models. The plot shown in figure 12 helps to visualise the structure of the network and better understand the two stream of inputs.



Note : Since we have used a pre-trained embedding layer, we had to freeze it (trainable = False), before training the model, so that it does not get updated during the backpropagation.

Image vector is nothing but the output of ResNet model we had used for encoding. We then add a Dense layer of 256 neurons to squeeze the output of ResNet from 2048 to 256. The LSTMs also give the hidden vector of size 256. Now both the outputs are concatenated and sent to a MLP i.e. Multilayer Perceptron. The MLP is just a decoder that predicts what should be the next word. The output layer has neurons equal to the vocabulary size. The output of the model is basically a probability distribution over the entire vocabulary. Since there are multiple outputs possible from the output layer, we have used categorical cross entropy as the loss function. The optimizer used to optimize the loss is the Adam optimizer.
