### Dimensionality-Reduction

The **Dimensionality Reduction Project** focuses on detecting distracted driving behaviors using a dataset of **22,424 colored images** from dashboard cameras. By applying image preprocessing, **Principal Component Analysis (PCA)**, Randomized PCA, and feature extraction techniques like DAISY, we reduce data complexity to enable effective classification of distracted behaviors. This project aims to enhance road safety and support better risk assessments.

---
### About the Dataset

**Domain**  
The dataset belongs to the domain of **automotive safety and computer vision**, aiming to detect distracted driver behaviors using machine learning and image classification.



**Dataset Overview**  
The dataset consists of **22,424 colored images** captured from dashboard cameras. Each image represents a driver exhibiting specific behaviors, categorized into one of ten classes. The goal is to train a model that can classify these behaviors accurately, contributing to road safety and insurance risk management.


**Attribute Information**  
- **Image Data**: Raw 2D RGB images (unprocessed) with various driver behaviors.  
- **Image Labels**: Each image is labeled according to one of ten distracted driving classes.  
- **Classes**:
  - **c0**: Normal driving  
  - **c1**: Texting - right  
  - **c2**: Talking on the phone - right  
  - **c3**: Texting - left  
  - **c4**: Talking on the phone - left  
  - **c5**: Operating the radio  
  - **c6**: Drinking  
  - **c7**: Reaching behind  
  - **c8**: Hair and makeup  
  - **c9**: Talking to passenger  


**Remaining Features**  
- **Image size**: All images are larger than 20x20 pixels and have not been pre-processed.  
- **File format**: Images are stored in directories, categorized by labels.  
- **Evaluation Metric**: Submissions are evaluated using **multi-class logarithmic loss**.


**Description**  
Distracted driving is a significant contributor to traffic accidents, leading to **425,000 injuries** and **3,000 fatalities annually** (CDC statistics). The State Farm dataset challenges participants to develop machine learning models that can **automatically detect driver behaviors** using dashboard camera images. 

By classifying behaviors such as texting, talking on the phone, or reaching behind, this dataset provides an opportunity to create predictive systems that enhance road safety and help insurance companies assess driver risks more effectively.

---
### Data Source:

- State Farm Distracted Driver Detection:
https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview
---

## Project Objectives  

### 1. Business Understanding  
- Address the critical issue of distracted driving, which contributes to **3,308 fatalities annually** according to NHTSA, and affects thousands of road users daily.  
- Use the **State Farm Distracted Driver Detection Dataset**, consisting of **22,424 colored images** of drivers engaging in different behaviors, captured via dashboard cameras.  
- Define the **prediction task** as classifying driver behaviors into 10 distinct categories, including safe driving, texting, talking on the phone, and other distractions.  
- Highlight the potential impact of this work, such as integrating accurate detection systems into advanced driver-assistance systems (ADAS) to enhance road safety and support insurance risk assessments.

---

### 2. Data Preparation  
- Read the raw images as numpy arrays.  
- Resize and recolor the images as needed to ensure uniform dimensions and structure.  
- Linearize images into 1-D arrays to facilitate exploratory analysis and machine learning model input.  
- Visualize sample images for each class to confirm dataset integrity and label accuracy.

---

### 3. Dimensionality Reduction  
- Apply **Principal Component Analysis (PCA)** to reduce the high dimensionality of the image data while retaining critical information.  
- Perform **Randomized PCA** to compare its performance and computational efficiency with standard PCA.  
- Visualize and analyze the explained variance for both methods to determine the optimal number of components needed for accurate representation.  

---

### 4. Feature Extraction  
- Extract meaningful image features using advanced techniques, such as:
  - **Gabor filters** for texture analysis,  
  - **Histogram of Oriented Gradients (HOG)** for edge-based features,  
  - **DAISY descriptors** for key point-based feature extraction.  
- Analyze extracted features to assess their effectiveness for classifying driver behaviors.

---

### 5. Modeling and Evaluation  
- Use a **nearest neighbor classifier** to evaluate the effectiveness of the extracted features.  
- Assess the performance of dimensionality reduction and feature extraction methods in supporting the prediction task.  
- Visualize and analyze feature differences across target classes using heatmaps, pairwise comparisons, or statistical summaries.  

---
## Tools and Libraries Used  

The following libraries and tools were used for data processing, analysis, and visualization:  

- **Python 3.11.9**: Programming language.  
- **Jupyter Notebook**: Interactive coding environment for exploratory analysis and visualization.  
- **Pandas**: Data manipulation and preprocessing.  
- **NumPy**: Numerical operations, including image array transformations.  
- **Matplotlib**: Data visualization and plotting.  
- **Seaborn**: Statistical data visualization for creating heatmaps and pairwise comparisons.  
- **OpenCV**: Image preprocessing, including resizing and recoloring tasks.  
- **Scikit-learn**: Dimensionality reduction using PCA and Randomized PCA, and model evaluation with nearest neighbor classifiers.  
- **Scikit-image**: Feature extraction using HOG and DAISY descriptors.  
---
## How to Run  

**1. Clone the Repository:**  
```markdown

```bash
git clone https://github.com/your-username/Dimensionality-Reduction.git
cd Dimensionality-Reduction
```

**2. Download the Dataset:**  
Use the Kaggle CLI to download the **State Farm Distracted Driver Detection** dataset:  
```bash
kaggle competitions download -c state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip -d data/
```

**3. Install Dependencies:**  
Install the required libraries using `pip`:  
```bash
pip install pandas matplotlib seaborn numpy opencv-python scikit-learn scikit-image
```

**4. Run the Notebook:**  

- **Open the Jupyter Notebook**:  
```bash
jupyter notebook Dimensionality_reduction.ipynb
```



