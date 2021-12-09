---
layout: post
title: Bahdanau and Luong Attention Models
subtitle: What is the difference between both models?
thumbnail-img: /assets/img/posts/2021-12-09-attention/attention_research_1.png
tags: [DL, attention, masters]
use_math: true
---
    
## Bahdanau vs Luong Attention

### Bahdanau Attention

The Bahdanau Attention model is an attention model that learns to align and translate jointly. It is also known as an additive attention as it performs a linear combination of encoder states and the decoder states. Next, we can see a bit more explanation about this attention model:

```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.w = tf.keras.layers.Dense(units)
        self.u = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.v(tf.nn.tanh(self.w(query_with_time_axis) + self.u(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        
        c = attention_weights * values
        context_vector = tf.reduce_sum(c, axis=1)

        return context_vector, attention_weights
```

- All hidden states of the encoder(forward and backward) and the decoder are used to generate the context vector.
- The attention mechanism aligns the input and output sequences, with an alignment score parameterized by a feed-forward network. It helps to pay attention to the most relevant information in the source sequence.
- The model predicts a target word based on the context vectors associated with the source position and the previously generated target words.


```python
display.Image("./BA.png")
```
    
![png](Rodolfo_Assignment_3_files/Rodolfo_Assignment_3_10_0.png)
    

These 3 steps explained above can be cleary seen in the BahdanauAttention class and the image above. In this example, the sequence given to the encoder was a sentence in English and the decoded result was the translation in Arab.


### Luong General Attention

The Luong Attention model is an attention model that has a multiplicative attention, instead of an additive attention like Bahdanau. It reduces encoder states and decoder state into attention scores by simple matrix multiplications. Simple matrix multiplication makes it is faster and more space-efficient. In this specific case of Luong General Attention, attention is placed on all source positions, unlike Local attention where attention is placed only on a small subset of the source positions per target word. Next, we can see a bit more explanation about this attention model:

```python
class LuongDotAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        # LUONGH Dot-product
        score = tf.transpose(tf.matmul(query_with_time_axis, 
                                       values_transposed), perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```

```python
class LuongGeneralAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongGeneralAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        #LUONGH General
        score = tf.transpose(tf.matmul(query_with_time_axis, self.W(values_transposed)), perm=[0, 2, 1])
        attention_weights = tf.nn.softmax(score, axis=1)

        c = attention_weights * values
        context_vector = tf.reduce_sum(c, axis=1)

        return context_vector, attention_weights
```

- In the decoding phase at each time step Luong Attention model first take the hidden state at the top layer of a stacking LSTM as an input.
- The goal is to derive a context vector to capture relevant source-side information to help predict the current target word.
- Context vectors are fed as inputs to the next time steps to inform the model about past alignment decisions.


```python
display.Image("./GA.png")
```

![png](Rodolfo_Assignment_3_files/Rodolfo_Assignment_3_17_0.png)
    

These 3 steps explained above can be cleary seen in the LuongGeneralAttention and LuongDotAttention classes and the image above. In comparison we can look at the next image which is the Local attention model.


```python
display.Image("./LA.png")
```

    
![png](Rodolfo_Assignment_3_files/Rodolfo_Assignment_3_19_0.png)
    
Now we can clearly see the difference of the Local vs Global Luong Attention models. 

