# CS4985 Capstone – LLM Pipeline Project

This project simulates building a full Large Language Model (LLM) pipeline from scratch. The goal is not to actually train a massive model, but to understand and demonstrate each stage of the workflow in a structured and realistic way.

For this project, we are focusing on sentiment analysis, where the model classifies text as positive or negative.

------------

## Pipeline Overview

The pipeline for this project follows these steps:

1. Data Collection  
2. Data Preprocessing  
3. Model Selection  
4. Fine-Tuning  
5. Evaluation  
6. Deployment  

Each stage is organized into its own folder in this repository.

------------

## Part 1: Data and Setup

For the data, we are using a labeled dataset where each piece of text has a sentiment label (positive or negative). This allows the model to learn patterns between the input text and the expected output.

Before training, the data is cleaned and formatted into input-output pairs.

Instead of building a model from scratch, we are using a pre-trained language model (such as a transformer from HuggingFace). This makes the process more efficient and realistic.

------------

## Part 2: Fine-Tuning

To adapt the model to our task, we use Supervised Fine-Tuning (SFT).

This means the model is trained on labeled examples so it can learn how to correctly classify sentiment. Over time, it improves its predictions based on the data.

------------

## Evaluation and Deployment

The model would be evaluated using accuracy and other performance metrics to make sure it works correctly.

Once validated, it could be deployed through an API so users can input text and receive predictions in real time.

------------

## Conclusion

This project demonstrates the full lifecycle of an LLM pipeline in a structured and practical way. While we are not building a full-scale model, the workflow reflects how real AI systems are developed and used.
