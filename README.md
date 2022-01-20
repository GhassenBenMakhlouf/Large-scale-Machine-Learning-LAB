# Learning for Small Data Sets: Classification of Sound Signals

## Problem
Engineers often diagnose broken vehicle components by their sound, e.g. the sound of the engine, the sound of the breaks or some rattling. Automatic detection and classification of mechanical noises caused by vehicle components helps engineers to repair cars andincreases safety by warning the driver if something is wrong. However, in real-world applications data sets are usually small and collecting data is expensive.

## Data
Real-world dataset from BMW that contains mechanical sounds of vehicle components and publicly available audio data sets.
<p align="center">
  <img align="right" src="./images/data_stats.png" alt="drawing" width="28%"/>
</p>

#### Dataset from BMW

* Real-world Brake noise dataset
* Contains 1090 samples over 6 classes
* Unbalanced, small and noisy Dataset  
  => our main Idea: use pretrained models  
  additionally: Cross validation & Data Augmentation

#### Data Preprocessing
<p align="center">
  <img align="right" src="./images/mel.png" alt="drawing" width="25%"/>
</p>

Original Audio Data (Waveform):
Sample Amplitude of sound wave at regular time intervals  
=> Transformation to Spectrogram:
from Time to Frequency Domain  
Output:
* Frequency Time Plot
* Image-like!

## Project Goals
<p align="center"> <img align="left" src="./images/robust.jpg" alt="drawing" width="3%"/></p>
* Implement robust classifiers. 
<p align="center"> <img align="left" src="./images/reduce.png" alt="drawing" width="3%"/> </p>
* Reduce the number of training samples per class. 
<p align="center"> <img align="left" src="./images/analyze.png" alt="drawing" width="3%"/> </p>
* Analyze and optimize the performance.

## Used Architectures
<p align="center">
  <img align="right" src="./images/panns.png" alt="drawing" width="30%"/>
</p>

### PANNs
* Input: both Spectrogram and Waveform
* Very deep architecture: Two **parallel** branches → Features Maps → Concatenation → Conv block → Classification
* Fully **pretrained** on the *AudioSet* dataset => achieves current state-of-the-art on AudioSet

### SoundCLR
* Input: Spectrogram
* ResNet50 pretrained on ImageNet
* Model is trained with a **Hybrid loss**. 
* **Hybrid loss: Weighted sum** of supervised contrastive loss and a cross-entropy loss.
<p align="center">
  <img src="./images/soundCLR.png" alt="drawing" width="50%"/>
</p>

### TALNetV3
* Input: Spectrogram
* Dual backbone consisting of:
  * Global feature extractor: **pretrained** on *AudioSet*
  * Specific feature extractor: **NOT** pretrained
<p align="center">
  <img src="./images/talnet.png" alt="drawing" width="50%"/>
</p>

## Results
<p align="center">
  <img align="right" src="./images/exp1.png" alt="drawing" width="30%"/>
</p>

### Results on the whole Dataset
First experiments:  
Training on the whole Dataset with & without Data Augmentation  
Metric used: Test Accuracy

Acquiring training samples takes a lot of effort!  
=> BMW interested in training with **fewer** samples per Class    
=> Next experiments: Investigate model performance on 40/20/10/5 samples **per Class**  
Metric: additionally Top 2/3 Accuracy

### Results on fewer Samples per Class
<table><tr>
<td> <img align="right" src="./images/plot1.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot2.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* PANNs achieves best results, 93.9% on 40 samples
* SoundCLR benefits from Augmentation but has problems with fewer samples
* TALNetV3 performs worse with Augmentation


<table><tr>
<td> <img align="right" src="./images/plot3.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot4.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* PANNs achieves over 93% Top 2 Accuracy for > 10 samples
* SoundCLR still has problems with fewer samples


<table><tr>
<td> <img align="right" src="./images/plot5.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot6.png" alt="Drawing" width="75%"/> </td>
</tr></table>

PANNs achieves:
* Over 94.5% Top 3 Accuracy even for 5 samples
* \>99% Top 3 Accuracy for 40 samples

### Class-wise Accuracies PANNs

<table><tr>
<td> <img align="right" src="./images/plot7.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/data_stats.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* “Hubknarzen” hardest class to learn, always below 90% accuracy
* “Schrummknarzen” & “No brake noise” accuracy drops significantly for fewer samples
* “Scheibenknacken” is the easiest class to learn

### Further Analysis PANNs 40 Samples
Accuracies only gave us an idea, whether the samples are predicted correctly or not. But we don’t have any idea about the wrongly predicted samples.

Two other metrics were computed:  
* class wise probabilities: averaged probabilities for each class
* confusion matrix: number of  predicted for each ground truth label

Why?
* to analyse further the effect of the number of samples on predicting each class and how confident the model is about its predictions 
* And to explore similarities between classes

<table><tr>
<td> <img align="right" src="./images/plot8.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot9.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* over 80% confidence for all classes except “Hubknarzen”
* a confusion between the three first classes. For example, almost 17% of samples of class “Hubknarzen” are predicted as “No Brake Noise”


<table><tr>
<td> <img align="right" src="./images/plot10.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot11.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* Here we can again see the drop in class “No brake noise” (confidance under 80%) 

<table><tr>
<td> <img align="right" src="./images/plot12.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot13.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* significant drop in confidence on “Schrummknarzen”
* overall drop


<table><tr>
<td> <img align="right" src="./images/plot14.png" alt="Drawing" width="75%"/> </td>
<td> <img align="left" src="./images/plot15.png" alt="Drawing" width="75%"/> </td>
</tr></table>

* “Scheibenknacken” (5th class) still almost always predicted correctly => easiest class
* “Quietschen” (4th class) also with good predictions. 
* However, the model has problems with the other classes.
* Worst performance with class Knarzen, with average probability is below 50%.

## Summary
Best Results with PANNs: 
* 96.6% test accuracy on full BMW dataset.
* 93.9% on 40 samples per class
* over 93% top 2 accuracy for up to 10 samples
 
=> It is possible to train a decent model using 10-20 samples per class.  
=> A fully pretrained architecture should be used for a dataset this small.

Performance with Augmentation depends on architecture:
* No significant boost for PANNs
* Accuracy increase for SoundCLR

 


