# %% [markdown]
# NOTE TO EDITORS: The following lines of code should be edited before use:
# LINE 137/138 - Set to your own directory.
# LINE 524 - Set to a folder directory name containing the images to be classified.
# LINE 534 - Set to the correct label for the images being classified, based on the list of ImageNet categories
# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
#  
#
#
#
# # ENGR415 — Lab Practice 2 (Python on Google Colab)
# 
# This notebook is a **beginner-friendly** Python version of Lab Practice 2.
# 
# ## What you will do
# - Use **three pretrained image classification models**: **AlexNet**, **VGG16**, and **GoogLeNet**
# - Classify a single image and view the **Top-5 predictions**
# - Compare the three models on the **same image**
# - Use a webcam (via the browser) to take photos and classify them
# 
# ## Important note about webcam in Colab
# Colab runs on a remote server. Continuous webcam video streaming is not always stable.
# This notebook uses a reliable method: **take a photo in the browser**, then classify the photo in Python.
# %%
# Import libraries
# CLASS LIST BTW: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
import torch, cv2, glob, time, os
import torch.nn.functional
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights, GoogLeNet_Weights, ResNet50_Weights, EfficientNet_V2_L_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale, resize
import logging # for debug purposes
logging.basicConfig(level=logging.INFO, format=' %(asctime)s -  %(levelname)s -  %(message)s') # Initialise logging
#from asyncio import create_eager_task_factory
#from sys import exception
#from pyparsing import null_debug_action
# Use CPU for Torch
device = "cpu"


def getColour(i, k):
    t = i / (k-1)  # normalising i based on number of classes found
    r = t
    g = 1 - t
    b = 0
    return (r, g, b)
    
modelLineColours = {
    "AlexNet": "blue",
    "VGG16": "orange",
    "GoogLeNet": "purple",
    "ResNet": "green",
    "EfficientNet": "pink"
}

# %% Model Initialisation (load pretrained models in prediction mode)
# Load pretrained models (AlexNet, VGG16, GoogLeNet) and set them to prediction mode.
# Load pretrained AlexNet
alexnet_weights = AlexNet_Weights.DEFAULT
alexnet = models.alexnet(weights=alexnet_weights).to(device)
alexnet.eval()
# Load pretrained VGG16
vgg_weights = VGG16_Weights.DEFAULT
vgg16 = models.vgg16(weights=vgg_weights).to(device)
vgg16.eval()
# Load pretrained GoogLeNet
googlenet_weights = GoogLeNet_Weights.DEFAULT
googlenet = models.googlenet(weights=googlenet_weights).to(device)
googlenet.eval()
# Load pretrained resnet
Resnet_weights = ResNet50_Weights.IMAGENET1K_V2
resnet = models.resnet50(weights=Resnet_weights).to(device)
resnet.eval()
# Load pretrained efficientnet
Efficientnetweights = EfficientNet_V2_L_Weights.DEFAULT
efficientnet = models.efficientnet_v2_l(weights=Efficientnetweights).to(device)
efficientnet.eval()

# %%
# Store models in a dictionary for easy switching
models_dict = {
    "AlexNet": (alexnet, alexnet_weights),
    "VGG16": (vgg16, vgg_weights),
    "GoogLeNet": (googlenet, googlenet_weights),
    "ResNet": (resnet,Resnet_weights),
    "EfficientNet": (efficientnet,Efficientnetweights) }
# Official preprocessing steps (resize, crop, normalize) for each model
preprocess_dict = {
    "AlexNet": alexnet_weights.transforms(),
    "VGG16": vgg_weights.transforms(),
    "GoogLeNet": googlenet_weights.transforms(),
    "ResNet": Resnet_weights.transforms(),
    "EfficientNet": Efficientnetweights.transforms() }
# Category names (1000 classes)
# The model outputs an index number; categories convert index -> readable label
categories = alexnet_weights.meta["categories"]
print("Number of categories:", len(categories))
print("Example category names:", categories[:10])

# %% Function Initialisation
# Image Preprocessing function
def imagePreprocessing(modelName, image):
    # Image formaatting and preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    preprocess = preprocess_dict[modelName]
    input_tensor = preprocess(img) # Preprocess the image (convert to model input format)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension: [channels, height, width] -> [1, channels, height, width]
    return img, input_batch

# Function to classify an image passed in, tested with a given model
def classify_image(modelName, image, top_k=5):
    model, _weights = models_dict[modelName] # Model acquisition
    img, input_batch = imagePreprocessing(modelName, image) # Image formatting/reprocessing

    # Predict without training-related computations
    with torch.no_grad():
        outputs = model(input_batch)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0) # Convert raw scores to probabilities (values between 0 and 1 that sum to 1)
    top_prob, top_index = torch.topk(probabilities, top_k) # Get Top-K results

    # Convert to readable results
    results = []
    for prob, index in zip(top_prob, top_index):
        label = categories[index.item()]
        results.append((label, float(prob.item())))
    print(results)
    return results


# Return image addresses using chosen input category and glob module
def getImages(category):
    # Fetch all images within the specified category folder with suffix .jpg or .jpeg
    files = sorted(glob.glob("FINALIMAGES/{}/*.jpg".format(category), recursive=True) + glob.glob("FINALIMAGES/{}/*.jpeg".format(category), recursive=True))
    #files = glob.glob("FINALIMAGES/trafficLight/tl10.jpeg") # SINGLE IMAGE TEST
    if len(files) == 0:
        raise ValueError("Category is empty")
    images = []
    logging.info("Fetched images: " + str(files))

    # Convert all fetched files to images for further manipulation
    for filename in files:
        img = cv2.imread(filename)
        if img is not None:
            images.append((img, filename))
    return(images)

# Add zero mean gaussian noise to selected image
def addNoise(image,strength):
    if image is None:
        raise Exception("Image load failed")
    noisyImage = image.astype(np.float32) + np.random.normal(0,strength,image.shape).astype(np.float32) # mean,std_dev,img.shape
    noisyImage = np.clip(noisyImage, 0,255).astype(np.uint8)

    # DISPLAY IMAGE (DEBUGGING)
    #image = rescale(image, 0.1, channel_axis=2, anti_aliasing=False)
    #noisyImage = rescale(noisyImage, 0.1, channel_axis=2, anti_aliasing=False)
    #cv2.imshow("Original image", image)
    #cv2.imshow("Noisy Image", noisyImage); cv2.waitKey(0); cv2.destroyAllWindows()
    return noisyImage

# Determine the saliency map; map of the gradients of the output class score relative to the pixels of the input image
# Pinpoints areas the network focussed on when making its prediction
# https://medium.com/@yalcinselcuk0/thrustworthy-machine-learning-gradient-decomposition-based-xai-saliency-maps-ebfe5fe4d430
def saliencyMap(modelName, image):
    model, _weights = models_dict[modelName] # Model acquisition
    img, input_batch = imagePreprocessing(modelName, image) # Image formatting/reprocessing

    # Track gradients
    input_batch.requires_grad_()
    # Compute logit outputs from model and obtain predicted class
    output = model(input_batch)
    classIndex = output.argmax(dim=1).item()
    score = output[0, classIndex]
    # Obtain gradients for saliency map
    model.zero_grad() 
    score.backward()

    # Get saliency map, resize to original image size, and normalise
    saliency, _ = torch.max(input_batch.grad.data.abs(), dim=1)  # max across channels
    saliency = saliency.squeeze().numpy()
    saliency = resize(saliency, (image.shape[0], image.shape[1]), preserve_range=True)
    normalisedSaliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Plot original image, saliency map and overlay
    plt.subplot(1, 3, 1)
    plt.imshow(img); plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(normalisedSaliency, cmap='jet'); plt.title("Saliency Map"); plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray'); plt.imshow(normalisedSaliency, cmap='jet', alpha=0.5); plt.title(f"Overlay"); plt.axis('off')
    
    plt.suptitle(f"{modelName} (predicted class {categories[classIndex]})")
    plt.tight_layout()
    plt.show()

    return normalisedSaliency

# Determine the gradient-weighted class activation map
# Highlights important region of the image
# https://medium.com/@bmuskan007/grad-cam-a-beginners-guide-adf68e80f4bb
def gradCAM(modelName, image, target_class=None):
    model, _weights = models_dict[modelName] # Model acquisition
    img, input_batch = imagePreprocessing(modelName, image) # Image formatting/reprocessing

    model.eval()

    # Storing forward activations and backwards gradients
    activations = [] # feature maps
    gradients = [] # gradients

    def saveGradient(grad):
        gradients.append(grad)

    # When target convolution layer runs
    # store the activation feature maps and
    # register a backwards hook to capture gradients during backpropagation
    def forward_hook(layer, input, output):
        activations.append(output) # save feature map
        # capture gradients when they backpropagate through the output tensor
        output.register_hook(saveGradient)

    # Select last convolution layer to generate activations (model dependent!)
    if modelName in ["VGG16", "AlexNet"]:
        target_layer = list(model.features.children())[-1]
    elif modelName == "ResNet":
        target_layer = model.layer4[-1]
    elif modelName == "EfficientNet":
        target_layer = model.features[-1][0]
    elif modelName == "GoogLeNet":
        target_layer = model.inception5b  # approximate
    else:
        raise ValueError("Model not supported for CAM")

    # Attach custom hook to the layer (when target_layer runs, forward_hook function also runs)
    # 
    handle = target_layer.register_forward_hook(forward_hook)

    # Compute logit outputs from model and obtain predicted class
    output = model(input_batch) # FORWARD PASS - ALSO RUNS FORWARD_HOOK
    classIndex = output.argmax(dim=1).item()
    score = output[0, classIndex]

    # Obtain gradients for CAM map
    model.zero_grad()
    score.backward() # Backpropagation, triggers SAVEGRADIENT

    # Fetch CAM data
    activations_tensor = activations[0] # [1, C, H, W] C=feature maps, H/W is spatial info
    gradients_tensor = gradients[0] # [1, C, H, W]
    # Perform pooling operation on gradients
    weights = torch.mean(gradients_tensor, dim=(2,3), keepdim=True) # [1,C,1,1] scalar weight per feature map --- WEIGHTS
    
    # Get CAM map, resize to original image size, and normalise
    cam = torch.sum(weights * activations_tensor, dim=1) # [1,H,W] i.e. collective impact of all feature maps, scaled by each feature map's importance (weights)
    cam = torch.relu(cam) # Eliminate negative influences
    cam = cam.squeeze().detach().cpu().numpy()
    cam = resize(cam, (image.shape[0], image.shape[1]), preserve_range=True)
    normalisedCAM = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Plot original image, saliency map and overlay
    plt.subplot(1, 3, 1) # LEFT PLOT; INPUT IMAGE
    plt.imshow(img); plt.title("Original Image"); plt.axis('off')
    plt.subplot(1, 3, 2) # MIDDLE PLOT; CAM MAP
    plt.imshow(normalisedCAM, cmap='jet'); plt.title("GradCAM Map"); plt.axis('off')
    plt.subplot(1, 3, 3) # RIGHT PLOT; OVERLAY
    plt.imshow(img, cmap='gray'); plt.imshow(normalisedCAM, cmap='jet', alpha=0.5); plt.title(f"Overlay"); plt.axis('off')
    
    plt.suptitle(f"{modelName} (predicted class {categories[classIndex]})")
    plt.tight_layout()
    plt.show()

    handle.remove()
    return normalisedCAM

# For a given image and chosen model, get the data relating to accuracy and noise
def getImageData(image,model, compareOutput):
    # Local level storage for results of a given image and model, at all noise levels tested.
    resultsDict = {}
    noiseLevels = [] # x-axis
    top1Classes = [] # top 1 prediction class labels
    top1Confidence = [] # top 1 prediction and y-axis for top 1 confidence
    confidencesInTop5 = [] # y-axis for confidence of correct class if in top 5, else 0
    colours = [] # plot dot colours (top 1 correct = green, top )
    times = [] # time taken to classify for given model
    top1Results = [] # For all noise levels, store whether top 1 prediction is correct (for top 1 accuracy)
    topKResults = [] # For all noise levels, store whether top 5 prediction is correct (for top 5 accuracy)

    # change this to just 0 to ignore noise testing (for general model performance comparison)
    #for i i range(0,1,25):
    for i in range(0,500,100): # change
        noiseStrength = i # useful for using log noise levels.
        noiseLevels.append(i) # Store noise level
        # Add noise if defined noise strength is not zero.
        if noiseStrength != 0:
            noisyImage = addNoise(image[0], noiseStrength)
        else:
            noisyImage = image[0]

        # Determine the top 5 results and store the time taken to classify
        top_k = 5
        t0 = time.time()
        noisyResults = classify_image(model, noisyImage, top_k)
        t1 = time.time()
        timeElapsed = t1-t0
        
        # Add all data to local storage
        times.append(timeElapsed) # times
        confidence = noisyResults[0][1] # [0] = top 1 result, [i] = confidence
        top1Classes.append(noisyResults[0][0]) # [0] = top 1 result, [i] = class label
        top1Confidence.append(confidence)
        top1Results.append(noisyResults[0][0] == compareOutput)
        for i in range(len(noisyResults)):
            if noisyResults[i][0] == compareOutput:
                confidencesInTop5.append(noisyResults[i][1])
                topKResults.append(True)
                colours.append(getColour(i, top_k))
                gradCAM(model, noisyImage)
                break
        else:
            confidencesInTop5.append(0)
            topKResults.append(False)
            colours.append((1,0,0))
        resultsDict[f"Noise {noiseStrength}"] = noisyResults[0]

    # THE FOLLOWING ARE USED FOR MODEL COMPARISON
    # Average time taken to classify a given image for a given model, across all noise levels
    avgTime = sum(times) / len(times)
    # Top 1 accuracy across all noise levels, for given image and model
    top1Accuracy = sum(top1Results) / len(top1Results) if top1Results else 0
    # Top 5 accuracy across all noise levels, for given image and model
    top5Accuracy = sum(topKResults) / len(topKResults) if topKResults else 0

    # PASS ALL RELEVANT DATA BACK TO MAIN LOOP: x-axis, y-axis (top 1), y-axis (top 5), colours, TTC, average accuracies
    return noiseLevels, top1Confidence, confidencesInTop5, colours, avgTime, top1Accuracy, top5Accuracy, top1Classes

# For each model, summated stores the summed time taken to classify, top 1 accuracy and top 5 accuracy across all noise levels.
# This will be divided by number of images tested to find the average
summatedData = {
    "AlexNet": [0,0,0],
    "VGG16": [0,0,0],
    "GoogLeNet": [0,0,0],
    "ResNet": [0,0,0],
    "EfficientNet": [0,0,0]
}

def topOnePlot(imageClassificationData, image, noiseLevels):
    for modelName, data in imageClassificationData.items():
        confidences = data[0]
        colours = []
        # Only consider red and green colours
        for i in range(len(data[2])):
            if data[2][i] == (0,1,0):
                colours.append("green")
            else:
                colours.append("red")
        plt.plot(noiseLevels, confidences, color=modelLineColours[modelName], label=modelName + " Top 1", linewidth=2)
        plt.scatter(noiseLevels, confidences, c=colours, s=40, edgecolor='black', zorder=5)
    plt.xlabel("Noise strength")
    plt.ylabel("Confidence of top 1 class")
    plt.title(f"Confidence vs Noise for image ({image[1]})")
    plt.legend()

def topFivePlot(imageClassificationData, image, noiseLevels):
    for modelName, data in imageClassificationData.items():
        confidences = data[1]
        colours = data[2]
        plt.plot(noiseLevels, confidences, color=modelLineColours[modelName], label=modelName + " Top 5", linewidth=2)
        plt.scatter(noiseLevels, confidences, c=colours, s=40, edgecolor='black', zorder=5)
    plt.xlabel("Noise strength")
    plt.ylabel("Confidence of correct class if in top 5, else 0")
    plt.title(f"Top 5 Confidence vs Noise for image ({image[1]})")
    plt.legend()

def topOneAccuracyChart(imageClassificationData, image):
    top1Accuracies = []
    modelNames = []
    for modelName, data in imageClassificationData.items():
        top1Accuracies.append(data[4])
        modelNames.append(modelName)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(modelNames, top1Accuracies)
    plt.bar_label(bars, labels=[f"{accuracy:.2f}" for accuracy in top1Accuracies], padding=3)
    plt.xlabel("Model")
    plt.ylabel("Top 1 Accuracy")
    plt.title(f"Top 1 Accuracy for image {image[1]}")
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

def topFiveAccuracyChart(imageClassificationData, image):
    top5Accuracies = []
    modelNames = []
    for modelName, data in imageClassificationData.items():
        top5Accuracies.append(data[5])
        modelNames.append(modelName)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(modelNames, top5Accuracies)
    plt.bar_label(bars, labels=[f"{accuracy:.2f}" for accuracy in top5Accuracies], padding=3)
    plt.xlabel("Model")
    plt.ylabel("Top 5 Accuracy")
    plt.title(f"Top 5 Accuracy for image {image[1]}")
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

def topAccuracyChart(imageClassificationData, image):
    top1Accuracies = []
    top5Accuracies = []
    modelNames = []
    for modelName, data in imageClassificationData.items():
        top1Accuracies.append(data[4])
        top5Accuracies.append(data[5])
        modelNames.append(modelName)

    top5Plot = np.array(top5Accuracies) - np.array(top1Accuracies)
    plt.figure(figsize=(8, 4))
    bars1 = plt.bar(modelNames, top1Accuracies, label="Top 1")
    bars2 = plt.bar(modelNames, top5Accuracies, label="Top 5", bottom=top1Accuracies)
    plt.bar_label(bars1, labels=[f"{accuracy:.2f}" for accuracy in top1Accuracies], padding=3)
    plt.bar_label(bars2, labels=[f"{accuracy:.2f}" for accuracy in top5Accuracies], padding=3)

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title(f"Top 1 and Top 5 Accuracy for image {image[1]}")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    
def averageOneAccuracyChart(summatedData, numberOfItems):
    averageTop1Accuracies = []
    modelNames = []
    for modelName, data in summatedData.items():
        averageTop1Accuracies.append(data[1] / numberOfItems)
        modelNames.append(modelName)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(modelNames, averageTop1Accuracies)
    plt.bar_label(bars, labels=[f"{accuracy:.2f}" for accuracy in averageTop1Accuracies], padding=3)
    plt.xlabel("Model")
    plt.ylabel("Average Top 1 Accuracy")
    plt.title(f"Average Top 1 Accuracy across {numberOfItems} images")
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

def averageFiveAccuracyChart(summatedData, numberOfItems):
    averageTop5Accuracies = []
    modelNames = []
    for modelName, data in summatedData.items():
        averageTop5Accuracies.append(data[2] / numberOfItems)
        modelNames.append(modelName)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(modelNames, averageTop5Accuracies)
    plt.bar_label(bars, labels=[f"{accuracy:.2f}" for accuracy in averageTop5Accuracies], padding=3)
    plt.xlabel("Model")
    plt.ylabel("Average Top 5 Accuracy")
    plt.title(f"Average Top 5 Accuracy across {numberOfItems} images")
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

def averageAccuracyChart(summatedData, numberOfItems):
    averageTop1Accuracies = []
    averageTop5Accuracies = []
    modelNames = []
    for modelName, data in summatedData.items():
        averageTop1Accuracies.append(data[1] / numberOfItems)
        averageTop5Accuracies.append(data[2] / numberOfItems)
        modelNames.append(modelName)

    top5Plot = np.array(averageTop5Accuracies) - np.array(averageTop1Accuracies)
    plt.figure(figsize=(8, 4))
    bars1 = plt.bar(modelNames, averageTop1Accuracies, label="Average Top 1")
    bars2 = plt.bar(modelNames, averageTop5Accuracies, label="Average Top 5", bottom=averageTop1Accuracies)
    plt.bar_label(bars1, labels=[f"{accuracy:.2f}" for accuracy in averageTop1Accuracies], padding=3)
    plt.bar_label(bars2, labels=[f"{accuracy:.2f}" for accuracy in averageTop5Accuracies], padding=3)

    plt.xlabel("Model")
    plt.ylabel("Average Accuracy")
    plt.title(f"Average Top 1 and Top 5 Accuracy across {numberOfItems} images")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

def averageTimeChart(summatedData, numberOfItems):
    averageTimes = []
    modelNames = []
    for modelName, data in summatedData.items():
        averageTimes.append(data[0] / numberOfItems)
        modelNames.append(modelName)
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(modelNames, averageTimes)
    plt.bar_label(bars, labels=[f"{time:.2f}s" for time in averageTimes], padding=3)
    plt.xlabel("Model")
    plt.ylabel("Average Time Taken to Classify (s)")
    plt.title(f"Average Time Taken to Classify across {numberOfItems} images")

def topPredictionsPerModel(topOneRecordings):
    for modelName, predictions in topOneRecordings.items():
        plt.figure(figsize=(10, 4))
        unique, counts = np.unique(predictions, return_counts=True)
        plt.bar(unique, counts, color=modelLineColours[modelName])
        plt.xlabel("Predicted Class")
        plt.ylabel("Count")
        plt.title(f"Top-1 Predicted Classes for {modelName} across all images")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

topOneRecordings = {
    "AlexNet": [],
    "VGG16": [],
    "GoogLeNet": [],
    "ResNet": [],
    "EfficientNet": []
}

numberOfItems = 0
#chosenModel = "VGG16"  # Change to "AlexNet" or "VGG16" if you want
for i in getImages("trafficLight"): # change
    numberOfItems += 1
    imageClassificationData = {
        "AlexNet": None,
        "VGG16": None,
        "GoogLeNet": None,
        "ResNet": None,
        "EfficientNet": None
    }

    compareOutput = "traffic light" # change
    for chosenModel in preprocess_dict:
        noiseLevels, top1Confidences, confidenceInTop5, colours, avgTime, top1Accuracy, top5Accuracy, top1Classes = getImageData(i, chosenModel, compareOutput)
        imageClassificationData[chosenModel] = (top1Confidences, confidenceInTop5, colours, avgTime, top1Accuracy, top5Accuracy, top1Classes)
        summatedData[chosenModel] = [summatedData[chosenModel][0] + avgTime, summatedData[chosenModel][1] + top1Accuracy, summatedData[chosenModel][2] + top5Accuracy]
        topOneRecordings[chosenModel].append(top1Classes[0])

    # Top 1 class and confidence for each image (zero noise), per model
    print(f"TOP 1 CLASS AND CONFIDENCE for image {i[1]}")
    for modelName, data in imageClassificationData.items():
        print(f"{modelName}: {data[6][0]}, {data[0][0]}")

    # Top 1 noise-confidence plot (1 image) for all models
    topOnePlot(imageClassificationData, i, noiseLevels)
    plt.show()

    # Top 5 noise-confidence plot (1 image) for all models
    topFivePlot(imageClassificationData, i, noiseLevels)
    plt.show()

    # Top 1 accuracy per model (1 image) for all models, bar chart plot
    topOneAccuracyChart(imageClassificationData, i)
    plt.show()

    # Top 5 accuracy per model (1 image) for all models, bar chart plot
    topFiveAccuracyChart(imageClassificationData, i)
    plt.show()

    topAccuracyChart(imageClassificationData, i)
    plt.show()

averageOneAccuracyChart(summatedData, numberOfItems)
plt.show()

averageFiveAccuracyChart(summatedData, numberOfItems)
plt.show()

averageAccuracyChart(summatedData, numberOfItems)
plt.show()

averageTimeChart(summatedData, numberOfItems)
plt.show()

topPredictionsPerModel(topOneRecordings)