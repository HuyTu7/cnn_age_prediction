# Facebook User's Age Prediction

### Description: 

Mined and investigated unstructured Facebook user’s posts with Vietnamese language processing with traditional ML methods, convolutional neural networks, and LSTM to predict the user’s age. Achieved accuracy = 76.4%. 

### Steps:

1. Mined unstructured FB user's posts (user's age need to be present)
2. Preprocessed to categorize age into classes (A: 18-23, B: 24-30, C: 30-40, D: 40+) 
3. Investigate class's distributrion and preprocess posts: <br /> 
        $\qquad$a) Replace emojis with " emoji_icon " to remove bias toward a specific emoji <br />
        b) Tokenize Vietnamese words <br />
        c) Remove Vietnamese stop-words  <br />
        d) Remove numbers and punctuations  <br />
        e) Collapse all posts into one vector 
4. Apply learners: <br />
        a) traditional machine learning model: <br />
            i. vectorize the words by frequency <br />
            ii. max absolute scaling <br />
            iii. apply SVM - accuracy: 50% <br />
        b) deep learning model: <br />
            i. only take the vector that is more than 200 items <br />
            ii. padding vectors up to 800 <br />
            iii. apply CNN model: <br />
                - embedding layer: 71% (I remember that the accuracy was higher - around 80 % - but cannot get the same one now)  <br />
                - word2vec (200 features and 15 contexts) :  60%


### Files:

```
├── text_analysis\
|   ├── data_processing.ipynb             
|   ├── text_analysis.py          
|   ├── cnn_age_predict.ipynb      
|   ├── utils\  
|       ├── utils.py
|       ├── smote.py
|       ├── class_weights.py
```

