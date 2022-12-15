# Roast or Toast – A Neural Network Model that detects Roast/Toast

A Deep Learning Network Model trained on thousands of comments from [Reddit](https://www.reddit.com/), more specifically from subreddits [r/RoastMe](https://www.reddit.com/r/RoastMe) (a subreddit devoted to roasting one another) and [r/ToastMe](https://www.reddit.com/r/ToastMe) (a subreddit devoted to motivating one another)

---

## Data Collection

The data, as mentioned above, was collected from the [r/RoastMe](https://www.reddit.com/r/RoastMe) and [r/ToastMe](https://www.reddit.com/r/ToastMe). And the wrapper used is [PRAW: The Python Reddit API Wrapper](https://praw.readthedocs.io/).

## Data Processing and Model Training

The comments were cleaned and broken down into sentences. Since the collected data was imbalanced between the two classes, Toast texts were undersampled to match the amount of Roast texts.

Texts were vectorized and fitted into a neural network model whose summary is as follows

```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 100, 100)          5098400   
                                                                
global_average_pooling1d (G  (None, 100)              0         
lobalAveragePooling1D)                                          
                                                                
dropout (Dropout)           (None, 100)               0         
                                                                
dense (Dense)               (None, 200)               20200     
                                                                
dropout_1 (Dropout)         (None, 200)               0         
                                                                
dense_1 (Dense)             (None, 300)               60300     
                                                                
dropout_2 (Dropout)         (None, 300)               0         
                                                                
dense_2 (Dense)             (None, 1)                 301       
                                                                
=================================================================
Total params: 5,179,201
Trainable params: 5,179,201
Non-trainable params: 0
_________________________________________________________________
```
The model scored 84.14% accuracy

## Model Implementation

The models were saved and now they are ready to get tested on real prompts.

Head over to [this Colab notebook](https://colab.research.google.com/drive/13rnnhbheW96ycWDNxZeq8oyF0cJKiuh5?usp=sharing) to test your own prompts!
