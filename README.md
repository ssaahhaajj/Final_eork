# PyTorch-API

## Introduction to the approach:

Intuition of this research is related to the research paper NeuralPower[1]. 

### Summary of NeuralPower[1]

Key insight:
All CNN architectures despite their variations consist of basic underlying primitives which exhibit similar execution characteristics per type of layer. Therefore, NeuralPower first models the power and runtime of the key layers that are commonly used in a CNN. Then, NeuralPower uses these models to predict the performance and runtime of the entire network. Below are the short points:

- Constructing model for both runtime and power layerwise.
- Distributed into three main layers convolutional, FC and pooling layer as they carry the main computation load.
- They proposed a learning based polynomial regression model to learn the coefficients for different layers which are assessed against the power and runtime measurements on different GPUs and DL software tools.
- The reasons for choosing above approach are :- in terms of accuracy polynomial models provide more flexibility and low prediction error, interpretability i.e. runtime and power have clear physical correlation with the layer’s configuration parameters and the last is the available amount of sampling data point
- To perform model selection ten-fold cross-validation is used along with lasso regularization.
<p align="center">
<img src="/images/method_neuralpower.png"  alt="methodology image">
</p>

### Our Approach:
We designed our model using the similar approach but for the transformer architectures such as BERT[2] etc. First, We collected the dataset on which we have to train our linear regression model. For collecting data I have built modules for fetching the FLOP count of each of the layer and the time required by each layer of transformer architectures.

We can profile **per layer time distribution** from the torchprof as follows:

```
import torchprof

with torchprof.Profile(model, use_cuda=True, want_op_file=False) as prof:
    model(input)
```
---
where,

**model**:  the model which needs to be profiled 

**input**:  the input to the model

**want_op_file**:  optional argument which is False by default, if we want output in file(csv format) then simply set ```want_op_file=True``` then calling prof.display() will store the output in the file named ```output_file.csv``` 

**use_cuda**:  optional argument which is False by default, if we want to use the cuda then simply set ```use_cuda=True```

We can profile from the modified thop for **FLOP** **count** as follows:

```
from thop import profile

flops, params = profile(model, inputs=(input_, ),want_op_file=False)
```
---
where,

**model**:  the model which needs to be profiled 

**input**:  the input to the model

**want_op_file**:  optional argument which is False by default, if we want output in file(csv format) then simply set ```want_op_file=True``` then calling prof.display() will store the output in the file named ```output_file.csv``` 

If we want a combined data collectively Then use below method:

```
from profile_f_t import profile
flops, params = profile(model, inputs=(input_, ),want_op_file=False)
```
---
where,

**model**:  the model which needs to be profiled 

**input**:  the input to the model

**want_op_file**:  optional argument which is False by default, if we want output in file(csv format) then simply set ```want_op_file=True``` then calling prof.display() will store the output in the file named ```output_file.csv``` 

Example for the above methods have been illustrated in the Google Colab modules [here](from profile_f_t import profile) and [here](https://colab.research.google.com/drive/1hHPaQOsaeyXmLv5ZfOL6uE-ToywCpyYy?usp=sharing).

The dataset has been prepared from the 40+ transformer models using various tokenizers and input data such as below combinations:
Model Name  | Tokenizer
------------- | -------------
OpenAIGPTLMHeadModel  | OpenAIGPTTokenizer 
OpenAIGPTDoubleHeadsModel  | OpenAIGPTTokenizer 
GPT2Model  | GPT2Tokenizer 
CTRLModel  | CTRLTokenizer 
TransfoXLModel  | TransfoXLTokenizer 
XLNetModel  | XLNetTokenizer 
XLMModel  | XLMTokenizer 
DistilBertModel  | DistilBertTokenizer 
RobertaModel  | RobertaTokenizer 
XLMRobertaModel  | XLMRobertaTokenizer 

> This repository contains only the code which was used for preparation of the dataset and can be used to calculate the FLOP Count and Time required by each of the layer along with the layer details like input size, outpt size, kernel size, etc. Other code will be published soon when our paper gets accepted.

## References

[1] Ermao Cai, Da-Cheng Juan, Dimitrios Stamoulis, Diana Marculescu, "NeuralPower: Predict and Deploy Energy-Efficient Convolutional Neural Networks", arXiv:1710.05420, 2017, [online] Available: https://arxiv.org/abs/1710.05420

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv:1810.04805, 2019, [online] Available: https://arxiv.org/abs/1810.04805
