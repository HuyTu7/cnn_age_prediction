# Facebook User's Age Prediction

### Description: 

Mined and investigated unstructured Facebook user’s posts with Vietnamese language processing while comparing traditional ML methods with deep learning methods such as convolutional neural networks, and LSTM to predict the user’s age. Achieved accuracy = 81.4%. 

### Dataset Overview:

Supervised classification learning problem. Text -> Age Category (A: 18-23, B: 24-30, C: 30-40, D: 40+)

Overall: 22694 entries  <br /> 
A: 5465 entries (24.08%) <br /> 
B: 7837 entries (34.53%) <br /> 
C: 3957 entries (17.44%) <br /> 
D: 896 entries (3.95%)

### Steps:

1. Mined unstructured FB user's posts (user's age need to be present)
2. Preprocessed to categorize age into classes () 
3. Investigate class's distributrion and preprocess posts: <br /> 
        <space> a) Replace emojis with " emoji_icon " to remove bias toward a specific emoji <br />
        <space> b) Tokenize Vietnamese words <br />
        c) Remove Vietnamese stop-words  <br />
        d) Remove numbers and punctuations  <br />
        e) Collapse all posts into one vector 
4. Apply learners: <br />
        a) traditional machine learning model: <br />
            &nbsp;&nbsp;&nbsp;&nbsp;i. vectorize the words by frequency <br />
            &nbsp;&nbsp;&nbsp;&nbsp;ii. max absolute scaling <br />
            &nbsp;&nbsp;&nbsp;&nbsp;iii. apply SVM - accuracy: 50% <br />
        b) deep learning model: <br />
            &nbsp;&nbsp;&nbsp;&nbsp;i. only take the vector that is more than 200 items <br />
            &nbsp;&nbsp;&nbsp;&nbsp;ii. padding vectors up to 800 <br />
            &nbsp;&nbsp;&nbsp;&nbsp;iii. apply CNN model: <br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- embedding layer: 71% (I remember that the accuracy back in the summer was higher - around 80 % - need to check again)  <br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- word2vec (200 features and 15 contexts) :  60%


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

