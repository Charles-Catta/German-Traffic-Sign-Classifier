## Project: Traffic Sign Classifier CNN
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

A convolutional neural network that classifies traffic signs with 95% accuracy.

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Running to code
The easiest way to run the Jupyter notebook is by using [Docker](https://store.docker.com/search?type=edition&offering=community)

Run the following to get started, this will pull a Docker container with all the required dependencies

```sh
git clone https://github.com/Charles-Catta/German-Traffic-Sign-Classifier.git

cd German-Traffic-Sign-Classifier
```
#### CPU only
```sh
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

#### With GPU support
Make sure that [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is installed
```sh
nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

Go to [localhost:8888](http://localhost:8888/) and insert the token printed to the console

---

### Quick links

[Code](https://nbviewer.jupyter.org/github/Charles-Catta/German-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb) (This is a jupyter notebook)

[Write up](https://htmlpreview.github.io/?https://github.com/Charles-Catta/German-Traffic-Sign-Classifier/blob/master/writeup.html)

---

## Example outputs
![softmax_prediction](https://github.com/Charles-Catta/German-Traffic-Sign-Classifier/raw/master/img/softmax_preds_0.png)
![softmax_prediction](https://github.com/Charles-Catta/German-Traffic-Sign-Classifier/raw/master/img/softmax_preds_1.png)
