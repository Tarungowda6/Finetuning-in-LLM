# Finetuning-in-LLM
PEFT Fine-tuning large language model (LLM)

PEFT Fine-tuning large language model (LLM)

By: Tarun S Gowda

Introduction:

A large language model (LLM) is a type of artificial intelligence (AI) program that can recognize
and generate text, among other tasks. LLMs are trained on huge sets of data— hence the name
"large." LLMs are built on machine learning specifically, a type of neural networking called a
transformer model.
In simpler terms, an LLM is a computer program that has been fed enough examples to be able to
recognize and interpret human language or other types of complex data. Many LLMs are trained
on data that has been gathered from the Internet — thousands or millions of gigabytes' worth of
text. But the quality of the samples impacts how well LLMs will learn natural language, so an
LLM's programmers may use a more curated data set.
LLMs use a type of machine learning called deep learning in order to understand how characters,
words, and sentences function together. Deep learning involves the probabilistic analysis of
unstructured data, which eventually enables the deep learning model to recognize distinctions
between pieces of content without human intervention.
LLMs are then further trained via tuning: they are fine-tuned or prompt-tuned to the particular task
that the programmer wants them to do, such as interpreting questions and generating responses, or
translating text from one language to another.
Large Language Models (LLMs) have dramatically transformed natural language processing
(NLP), excelling in tasks like text generation, translation, summarization, and question-answering.
However, these models may not always be ideal for specific domains or tasks.
To address this, fine-tuning is performed. Fine-tuning customizes pre-trained LLMs to better suit
specialized applications by refining the model on smaller, task-specific
datasets. This allows the model to enhance its performance while retaining its broad language proficiency

PEFT (Pretraining-Evaluation Fine-Tuning)

What is parameter-efficient fine-tuning (PEFT)?

Parameter-efficient fine-tuning (PEFT) is a method of improving the performance of pretrained
Large Language Model (LLM) and neural network for specific tasks or data sets. By training a
small set of parameters and preserving most of the large pretrained model’s structure, PEFT saves
time and computational resources.

How does parameter-efficient fine-tuning work?

PEFT works by freezing most of the pretrained language model’s parameters and layers while
adding a few trainable parameters, known as adapters, to the final layers for predetermined
downstream tasks.

The fine-tuned models retain all the learning gained during training while specializing in their
respective downstream tasks. Many PEFT methods further enhance efficiency with gradient
checkpointing, a memory-saving technique that helps models learn without storing as much
information at once.

Why is parameter-efficient fine-tuning important?

Parameter-efficient fine-tuning balances efficiency and performance to help organizations
maximize computational resources while minimizing storage costs. When tuned with PEFT
methods, transformer-based models such as GPT-3, LLaMA and BERT can use all the knowledge
contained in their pretraining parameters while performing better than they otherwise would
without fine-tuning.

PEFT is often used during transfer learning, where models trained in one task are applied to a
second related task. For example, a model trained in image classification might be put to work on
object detection. If a base model is too large to completely retrain or if the new task is different
from the original, PEFT can be an ideal solution.

PEFT Fine-Tuning Project

Welcome to the PEFT (Pretraining-Evaluation Fine-Tuning) project! This project which is
done by me Tarun S Gowda, focuses on efficiently fine-tuning large language models
using LoRA and Hugging Face's transformers library.
For this project, I will be using Jupyter Notebook as the main source of computing format

1. Efficiently train Large Language Models with LoRA and Hugging
Face

LoRA:

LoRA, a technique that accelerates the fine-tuning of large models while consuming less
memory.
To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates
with two smaller matrices (called update matrices) through low-rank decomposition.
These new matrices can be trained to adapt to the new data while keeping the overall
number of changes low. The original weight matrix remains frozen and doesn’t receive any
further adjustments. To produce the final results, both the original and the adapted
weights are combined.

About Hugging face:

Hugging Face is a machine learning and data science platform and community that
helps users build, deploy and train machine learning models.
It provides the infrastructure to demo, run and deploy artificial intelligence in live
applications. Users can also browse through models and data sets that other people
have uploaded. Hugging Face is often called the GitHub of machine learning because
it lets developers share and test their work openly.
The platform is important because of its open-source nature and deployment tools. It
allows users to share resources, models and research and to reduce model training
time, resource consumption and environment impact of AI development.
Hugging Face Inc. is the American company that created the Hugging Face platform.
The company was founded in New York City in 2016 by French entrepreneurs Clément
Delangue, Julien Chaumond and Thomas Wolf. The company originally developed a
chatbot app by the same name for teenagers. The company switched its focus to being
a machine learning platform after open sourcing the model behind the chatbot app

1. Setup Development Environment
   
In this example, we use the PyTorch Deep Learning AMI with already set up CUDA drivers and
PyTorch installed. We still have to install the Hugging Face Libraries, including transformers and
datasets. Running the following cell will install all the required packages

