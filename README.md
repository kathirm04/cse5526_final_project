# cse5526_final_project

# Chest X-Ray Condition Detection — CSE 5526 Final Project

Team: Matthew Weikel, Kathir Maarikarthykeyan
Dataset: CheXpert v1.0-small 

# Overview
This project implements automated chest X-ray disease classification using deep learning.

We evaluate two models:

Baseline: Fine-tuned DINOv3 transformer backbone

Advanced Approach (Submitted Model): ResNet-50 + Multi-Layer Perceptron (MLP) classifier head

The repository includes:

The trained ResNet-50 + MLP model (res_net_50_mlp.pth)

Inference script that runs on sample images

10 test example chest X-rays for demonstration

Instructions for installing dependencies and running inference

# Installation & Setup

## 1. Create a virtual environment
python3 -m venv venv

source venv/bin/activate (mac/Linux)
. .\venv\Scripts\activate (Windows)

## 2. Install dependencies
pip install -r requirements.txt

# Run model
## 1. run inference on a single image
python3 inference.py --model models/res_net_50_mlp.pth --image test_examples/example_1.jpg

## 2. run inference on all images
python3 inference.py

## 3. Output
A CSV file called predictions.csv

Each row contains:
- Image name
- Probability score for each of the 14 CheXpert disease labels
