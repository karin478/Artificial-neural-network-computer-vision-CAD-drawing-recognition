# Project Introduction

The goal of this project is to develop a neural network capable of image recognition. Analysis of the dataset revealed that images and annotation information are primarily located in the `image` folder and `label` files. Initial tests using a simple Convolutional Neural Network (CNN) indicated that it is more suited for overall image classification rather than identifying specific parts within an image.

Further research led us to consider fine-tuning already trained models to improve project outcomes. The models explored include:

1. **Faster R-CNN**: High accuracy with region proposal network, but slower speed.
2. **YOLO (You Only Look Once)**: Supports real-time processing, though slightly less accurate than Faster R-CNN, it enables global detection.
3. **SSD (Single Shot MultiBox Detector)**: Achieves a good balance between speed and accuracy, utilizing multiple feature maps.
4. **Mask R-CNN**: Extends Faster R-CNN by adding object segmentation for higher accuracy.

Ultimately, we selected the **YOLO** model for the following reasons:

- YOLO supports real-time processing, which aids in real-time blueprint recognition on devices like smartphones or VR equipment.
- Our dataset format conforms to the YOLO standard format.

## Training environment and hardware configuration
The focus of this project is to showcase ideas and explain behaviors, so Google Colab was chosen as the environment for model training and testing. In terms of hardware, NVIDIA T4 graphics card was used in small-scale training, while V100 and A100 graphics cards were used for training after scaling up data size and epochs.

## Model Version Selection

Initially, we opted for the **YOLO v5** version due to its extensive support library and demonstrated higher accuracy in numerous studies. As an alternative, we also considered the **YOLO v8** version.

## Model Training and Testing

### Prototype Test
‌‌‌This training is the first test, no additional data preprocessing was done, just a simple division. Here is the test report for the first prototype:
[Preliminary Test](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project.ipynb)
<details>
  <summary>Report</summary>
 
## Introduction

In the field of architectural design and analysis, identifying and cataloging elements like rooms, windows, and doors is crucial. The project employs the YOLOv5 algorithm, noted for its speed and accuracy in image recognition tasks. The primary objective is to evaluate the algorithm's performance in recognizing architectural components directly from drawings.

## Data

The dataset consists of 537 architectural drawings, encompassing a total of 14,635 labeled instances across three distinct categories: room, window, and door. The performance of the model on this dataset is summarized below:

| Class   | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---------|--------|-----------|-----------|--------|-------|----------|
| All     | 537    | 14,635    | 0.879     | 0.720  | 0.781 | 0.560    |
| Room    | 537    | 5,404     | 0.909     | 0.773  | 0.820 | 0.592    |
| Window  | 537    | 3,733     | 0.871     | 0.691  | 0.761 | 0.463    |
| Door    | 537    | 5,498     | 0.858     | 0.695  | 0.760 | 0.500    |



## Methodology

The study utilized the YOLOv5 model in a straightforward approach, applying no data preprocessing to assess baseline capabilities in architectural element detection. The dataset was divided into training and validation sets, with the model training until performance metrics plateaued.

## To be researched

The experiment results are from the training on the validation set using val.py alone. The results in the training summary are significantly higher than this data. However, because there were many issues with the images (such as not undergoing sRGB conversion), it may have led to skipping many non-compliant results during training evaluation, thereby improving the model performance. (Speculation) The above data was obtained when using the default 32 batch size for test set detection settings. As the performance of later training summaries gradually overlapped with val.py, and even when detecting with a smaller batch size like 8 batch size, a mAP50 of 0.927 was achieved by va.lpy's detection results. How to choose the batch size for val period has become a topic that can be studied. Therefore, in subsequent experiments, we will first use the training summary report as the main source of data.

## Results

The YOLOv5 model exhibited robust performance, particularly in identifying room categories. The precision and mAP scores were highest for rooms, suggesting the model's effectiveness in this context. While the performance on windows and doors was slightly lower, it still indicated considerable success in recognizing these elements within architectural drawings.

</details>

### Colors Enhancement Test
‌‌In the second test, we attempted data preprocessing because rotating images would cause the labels to lose coordinates. Therefore, we chose to only enhance colors and contrast in an attempt to improve the quality of model training. Here is the test report:
[Colors Enhancement Test](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_data_pre_process.ipynb)
<details>
  <summary>Colors Enhancement Report</summary>

# Data Preprocessing Impact Analysis on Model Performance

## Overview

This report provides a comparative analysis between the original and preprocessed datasets used in training a deep learning model for object detection. The goal is to evaluate the impact of data preprocessing on the model's performance.


## Data Preprocessing Techniques

In the development of our object detection model, specific data preprocessing techniques were employed to ensure the model's robustness and adaptability to varying input conditions. The following list outlines the techniques applied:

1. **Conversion of Grayscale Images to RGB**: To maintain consistency in input data format, all grayscale images are converted to RGB format. This is crucial as the model is designed to process three-channel RGB images.

2. **Conversion from RGBA to RGB**: Images in RGBA format, containing an alpha channel for transparency, are converted to standard RGB format. This standardization is important to avoid discrepancies in image formats and ensure uniform input to the model.

3. **Image Standardization**: Prior to augmentation, images are standardized to the `uint8` format. This standardization is necessary to align with the expected input format of the augmentation library and maintain consistency across the dataset.

4. **Hue, Saturation, Value Adjustments**: To introduce variability in the dataset and simulate different lighting conditions, the hue, saturation, and value of the images are randomly adjusted. This variability helps in enhancing the model's ability to generalize across different environmental settings.

5. **Random Brightness and Contrast Adjustments**: The model's adaptability to different lighting conditions is further improved by randomly adjusting the brightness and contrast of the images. This step ensures that the model can perform well under various lighting conditions, enhancing its practical applicability.

These preprocessing steps are integral to the training process, enhancing the model's performance and ensuring its effectiveness in real-world scenarios.

## Performance Metrics Comparison

### Overall Performance:

| Metric    | Before | After |
|-----------|--------|-------|
| Precision (P) | 0.879  | 0.882 |
| Recall (R)    | 0.72   | 0.839 |
| mAP50         | 0.781  | 0.901 |
| mAP50-95      | 0.56   | 0.643 |


### Performance by Class:

#### Room:

| Metric     | Before | After |
|------------|--------|-------|
| Precision  | 0.909  | 0.909 |
| Recall     | 0.773  | 0.892 |
| mAP50      | 0.82   | 0.941 |
| mAP50-95   | 0.592  | 0.679 |

#### Window:

| Metric     | Before | After |
|------------|--------|-------|
| Precision  | 0.871  | 0.88  |
| Recall     | 0.691  | 0.803 |
| mAP50      | 0.761  | 0.881 |
| mAP50-95   | 0.463  | 0.53  |

#### Door:

| Metric     | Before | After |
|------------|--------|-------|
| Precision  | 0.858  | 0.858 |
| Recall     | 0.695  | 0.821 |
| mAP50      | 0.764  | 0.881 |
| mAP50-95   | 0.625  | 0.72  |

![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/dd94229f-c5d2-4fef-af53-0a1737e42c3e)


## Analysis

### Overall Impact:

After preprocessing, the model's overall performance has seen notable improvements, particularly in terms of Recall and Mean Average Precision (mAP). These improvements suggest that preprocessing helps the model generalize better and more effectively recognize different object classes.

### Performance Variations by Class:

- The **Room** category showed the most significant performance improvement, especially in Recall and mAP50, indicating that the preprocessed model is more accurate in detecting more rooms.
- The **Window** and **Door** categories also showed performance improvements, especially in Recall, indicating that after preprocessing, the model has a higher coverage in detecting windows and doors.

### Influencing Factors:

The preprocessing steps include color space adjustments, and brightness and contrast adjustments. These improvements may have helped the model better distinguish between different object features, particularly under varying lighting and background conditions. The changes in color and contrast seem to aid in improving the model's ability to recognize different object categories.

## Conclusion

The data preprocessing has significantly impacted the model's performance positively, especially in terms of Recall and mAP metrics. This indicates that preprocessing steps like color adjustments and brightness/contrast adjustments are effective in enhancing the model's generalization ability in real-world scenarios. The specific improvements in recognizing certain object categories, such as rooms, windows, and doors, suggest these preprocessing techniques are particularly useful in enhancing the model's ability to detect specific objects. Further experimentation, such as different types of image enhancements, could be beneficial to determine the optimal data preprocessing workflow.

</details>

### Image Size Test
‌‌‌‌In the third test, we tested the impact of different image sizes on the model performance. The comparison options were 640 and 1280. Below is the test report:
[Image Size Test](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_image_size_test.ipynb)
<details>
  <summary>Image Size Report</summary>

# Model Performance Comparison Report: 640 vs 1280 Image Sizes

## Introduction

This report provides a comparative analysis between two different image sizes, 640 and 1280, used in training a deep learning object detection model. The objective is to assess the impact of image size on model performance metrics including Precision, Recall, mAP50, and mAP50-95.

## Overall Performance Comparison

| Metric     | Image Size 640 | Image Size 1280 |
|------------|----------------|-----------------|
| Precision  | 0.882          | 0.87            |
| Recall     | 0.839          | 0.791           |
| mAP50      | 0.901          | 0.852           |
| mAP50-95   | 0.643          | 0.614           |

## Performance by Class

### Room

| Metric     | Image Size 640 | Image Size 1280 |
|------------|----------------|-----------------|
| Precision  | 0.909          | 0.909           |
| Recall     | 0.892          | 0.85            |
| mAP50      | 0.941          | 0.896           |
| mAP50-95   | 0.679          | 0.654           |

### Window

| Metric     | Image Size 640 | Image Size 1280 |
|------------|----------------|-----------------|
| Precision  | 0.88           | 0.858           |
| Recall     | 0.803          | 0.759           |
| mAP50      | 0.881          | 0.831           |
| mAP50-95   | 0.53           | 0.511           |

### Door

| Metric     | Image Size 640 | Image Size 1280 |
|------------|----------------|-----------------|
| Precision  | 0.858          | 0.842           |
| Recall     | 0.821          | 0.763           |
| mAP50      | 0.881          | 0.828           |
| mAP50-95   | 0.72           | 0.677           |

![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/b8a47449-ec0d-4c04-9162-7eff846a0d53)


## Analysis

Upon comparing the two image sizes, it is evident that the model trained with 640 image size generally outperforms the one trained with 1280 image size across all main performance metrics. Specifically:

- The **Precision** sees a slight drop from 640 to 1280, indicating a marginal decrease in the proportion of true positive detections.
- The **Recall** metric shows a more significant decrease, suggesting that the model with 1280 image size is less capable of identifying all relevant instances in the dataset.
- **mAP50** and **mAP50-95** both decrease as the image size increases, which implies that the model's ability to accurately detect and localize objects diminishes with larger image sizes.

### Considerations

- **Computational Load**: The increased image size leads to higher computational requirements and longer inference times, which might not be justifiable given the decrease in performance metrics.
- **Data Representation**: Larger image sizes could introduce more complexity and variability that the current model architecture or training regimen may not handle optimally.
- **Optimization and Tuning**: The model might require different tuning or a different architecture to fully leverage the higher resolution provided by the 1280 image size.

## Conclusion

The comparative analysis between the 640 and 1280 image sizes demonstrates that, for this specific object detection model, a smaller image size of 640 provides better performance across several key metrics. While larger image sizes can theoretically offer more detailed information for object detection, they also pose greater challenges for model training and may require more computational resources. Future work should focus on optimizing model parameters and architectures to better accommodate larger image sizes or consider the trade-offs between resolution and performance for their specific application requirements.

  
</details>

### Rotating Image Test
‌‌‌‌‌In the fourth test, we attempted to rotate the images. We built a data pipeline using Python that can automatically generate three additional rotation angles for each image in the "image" folder and create corresponding new label files by calculating the coordinates of labels. Below is the report for this test:
[Rotating Image Test](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_image_rotation.ipynb)
<details>
  <summary>Rotating Image Report</summary>

## Evaluation Report: Image Rotation Preprocessing

This evaluation report analyzes the impact of image rotation preprocessing on the performance of object detection models. We compare the performance metrics of two models: one trained without image rotation preprocessing and another with such preprocessing.

### Evaluation Metrics

The following metrics were used for comparison:

- **Precision (P)**: The ability of the model to identify only relevant objects.
- **Recall (R)**: The ability of the model to find all relevant instances.
- **mAP50**: Mean Average Precision at an IoU (Intersection over Union) threshold of 0.5.
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.

### Results

The performance of the models on different classes (Room, Window, Door) is shown in the tables below:

#### Overall Performance

| Metric      | Without Rotation | With Rotation |
|-------------|------------------|---------------|
| Precision   | 0.88             | 0.887         |
| Recall      | 0.76             | 0.84          |
| mAP50       | 0.82             | 0.904         |
| mAP50-95    | 0.586            | 0.625         |

#### Performance by Class

##### Room

| Metric    | Without Rotation | With Rotation |
|-----------|------------------|---------------|
| Precision | 0.904            | 0.93          |
| Recall    | 0.807            | 0.89          |
| mAP50     | 0.858            | 0.946         |
| mAP50-95  | 0.62             | 0.659         |

##### Window

| Metric    | Without Rotation | With Rotation |
|-----------|------------------|---------------|
| Precision | 0.881            | 0.859         |
| Recall    | 0.729            | 0.807         |
| mAP50     | 0.801            | 0.878         |
| mAP50-95  | 0.483            | 0.495         |

##### Door

| Metric    | Without Rotation | With Rotation |
|-----------|------------------|---------------|
| Precision | 0.854            | 0.871         |
| Recall    | 0.745            | 0.822         |
| mAP50     | 0.803            | 0.889         |
| mAP50-95  | 0.655            | 0.721         |

![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/e175292c-ea2f-40f9-8c5f-1536315a6381)


### Conclusion

The comparison indicates that image rotation preprocessing improves the model's performance across all considered metrics. Notably, the improvement in Recall suggests that the model trained with rotation is better at identifying relevant instances across various orientations, enhancing its robustness and generalization capability. Therefore, incorporating image rotation into the data preprocessing steps is recommended for this object detection task, especially for applications requiring detection from multiple angles.

  
</details>

### Normalization test
[Normalization](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_Normalization%20test.ipynb)
<details>
  <summary>Normalization Test Report</summary>
## Comparison Evaluation Report on the Impact of Normalization in YOLO Model Performance

### Introduction
This report presents a comparative evaluation of the impact of normalization on the performance of a YOLO (You Only Look Once) model trained for object detection tasks. Two sets of results are analyzed: one with normalization applied during the training process and one without. The objective is to assess how normalization affects the model's precision (P), recall (R), mean Average Precision (mAP) at Intersection over Union (IoU) threshold of 0.50 (mAP50), and mAP at IoU thresholds from 0.50 to 0.95 (mAP50-95).


### Metrics Definitions
- **Precision (P)**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall (R)**: The ratio of correctly predicted positive observations to all observations in actual class.
- **mAP50**: The mean Average Precision calculated at an IoU threshold of 0.50.
- **mAP50-95**: The mean Average Precision averaged over IoU thresholds from 0.50 to 0.95.

### Results

#### Overall Performance
| Metric     | With Normalization | Without Normalization |
|------------|--------------------|-----------------------|
| Precision  | 0.863              | 0.887                 |
| Recall     | 0.830              | 0.840                 |
| mAP50      | 0.885              | 0.904                 |
| mAP50-95   | 0.601              | 0.625                 |

#### Performance by Class: Room
| Metric     | With Normalization | Without Normalization |
|------------|--------------------|-----------------------|
| Precision  | 0.913              | 0.930                 |
| Recall     | 0.882              | 0.890                 |
| mAP50      | 0.934              | 0.946                 |
| mAP50-95   | 0.636              | 0.659                 |

#### Performance by Class: Window
| Metric     | With Normalization | Without Normalization |
|------------|--------------------|-----------------------|
| Precision  | 0.824              | 0.859                 |
| Recall     | 0.796              | 0.807                 |
| mAP50      | 0.849              | 0.878                 |
| mAP50-95   | 0.465              | 0.495                 |

#### Performance by Class: Door
| Metric     | With Normalization | Without Normalization |
|------------|--------------------|-----------------------|
| Precision  | 0.850              | 0.871                 |
| Recall     | 0.811              | 0.822                 |
| mAP50      | 0.872              | 0.889                 |
| mAP50-95   | 0.703              | 0.721                 |

![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/292f5771-67bb-4df5-a2ca-014021fc2214)


### Discussion
The results demonstrate a nuanced impact of normalization on the YOLO model's performance. While normalization typically improves model training and convergence, in this case, the model without normalization exhibits higher performance across all metrics.

Notably, the model without normalization shows improvements in both precision and recall across all classes. This is particularly evident in the 'room' and 'door' categories, which see the most significant increases in both precision and recall, leading to higher mAP scores.

However, it's essential to consider that these results could be influenced by factors such as the specific dataset, the distribution of classes, and the training setup. Normally, normalization is expected to help the model generalize better and train faster, but these benefits might not be as pronounced depending on the specific circumstances of the training process and data characteristics.

### Conclusion
The comparative evaluation between the normalized and non-normalized YOLO models indicates that, in this instance, normalization does not enhance performance. In fact, the model trained without normalization outperforms its counterpart across all evaluated metrics. This finding suggests that while normalization is a valuable technique in many scenarios, its effectiveness can vary based on specific model configurations, data, and training conditions. Therefore, it's crucial to evaluate the impact of normalization within the context of each unique application.

</details>


### YOLO v5 small & YOLO v5 extra large & YOLO v8 extra large Experiment
Usually, larger and newer models tend to have better model performance. However, there may be different answers for specific problems. Here we compared and tested three models of YOLO: YOLO v5 for small size, YOLO v8 for extra-large size, and YOLO v8 for super large size. If time permits, we are likely to directly use the super large model of YOLOv8 as the benchmark for initial testing. However, the model of YOLO v5 has a longer history and stable performance. So it may be more suitable for enterprise solutions. Here we used the same parameters: image 640 and 30 epochs (from the model loss and mAP50 can see that all three models of YOLO start to converge from 20-30 epochs with oscillations; this oscillation is speculated to be caused by a preset high learning rate).
[YOLO v8 Experiment](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_YOLO8.ipynb)
<details>
  <summary>YOLO Size And Version Report</summary>

# Performance Comparison among YOLOv5, YOLOv5 Small, and YOLOv8

This table summarizes the performance metrics for the largest models of YOLOv5, YOLOv5 Small, and YOLOv8 across different categories:

| Metric               | YOLOv5 Large  | YOLOv5 Small  | YOLOv8        |
| -------------------- | ------------- | ------------- | ------------- |
| Number of Layers     | 322           | 157           | 268           |
| Number of Parameters | 86,186,872    | 7,018,216     | 68,126,457    |
| GFLOPs               | 203.8         | 15.8          | 257.4         |
| Overall Precision (P)| 0.873         | 0.88          | 0.87          |
| Overall Recall (R)   | 0.847         | 0.84          | 0.844         |
| Overall mAP50        | 0.902         | 0.905         | 0.905         |
| Overall mAP50-95     | 0.626         | 0.628         | 0.638         |
| Room Precision (P)   | 0.926         | 0.918         | 0.924         |
| Room Recall (R)      | 0.896         | 0.895         | 0.89          |
| Room mAP50           | 0.947         | 0.946         | 0.946         |
| Room mAP50-95        | 0.663         | 0.661         | 0.666         |
| Window Precision (P) | 0.839         | 0.853         | 0.85          |
| Window Recall (R)    | 0.82          | 0.816         | 0.804         |
| Window mAP50         | 0.871         | 0.879         | 0.876         |
| Window mAP50-95      | 0.498         | 0.501         | 0.51          |
| Door Precision (P)   | 0.854         | 0.868         | 0.835         |
| Door Recall (R)      | 0.825         | 0.81          | 0.837         |
| Door mAP50           | 0.886         | 0.89          | 0.894         |
| Door mAP50-95        | 0.718         | 0.723         | 0.739         |

![yolo_performance_subset_1](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/2f48b07c-6032-4aaa-83e1-8fe6d39d64fe)
![yolo_performance_subset_2](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/f485a31e-ce60-440a-b41c-edcff4d4d2b4)
![yolo_performance_subset_3](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/75238c0b-417d-406b-bb6e-19de556019fb)
![yolo_performance_subset_4](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/1f422f1e-4421-421b-b586-1529bfb8dbef)






From the updated comparison table, we can observe the following:

1. **Model Complexity and Computational Cost**: The small version of YOLOv5 offers a significant reduction in both the number of layers and parameters compared to its larger counterpart and YOLOv8. It also requires substantially less computational power (15.8 GFLOPs).

2. **Performance (Precision, Recall, mAP)**:
   - The small version of YOLOv5 shows competitive performance compared to the larger models, particularly in overall precision and mAP50.
   - Despite its smaller size, YOLOv5 Small performs comparably to or slightly better than YOLOv5 Large in several metrics and is competitive with YOLOv8, especially considering its lower computational requirements.

This data suggests that YOLOv5 Small could be a highly efficient model for environments with stringent computational or storage limitations while still maintaining high levels of accuracy.
However, one possibility that cannot be ruled out is that YOLO large size and updated models may have greater potential for hyperparameter tuning. If time permits, it is hoped to conduct more comprehensive testing on the large size and new architecture.

  
</details>

### Hyperparameter Evolutionary Algorithm Testing Experiment
‌‌‌‌‌‌‌In the this experiment, we began to try hyperparameter tuning. In traditional hyperparameter tuning, researchers often use the algorithm called grid search. However, this search method carries the risk of missing points between grids. After research, it was found that YOLO v5 supports evolutionary algorithm for hyperparameter optimization. Therefore, it was decided to use evolutionary algorithm for hyperparameter optimization in this experiment. Although we had to interrupt the hyperparameter search early due to time constraints, we still achieved a slight improvement in model performance. If we have enough time for training, it is expected that the improvement will be very significant. The following is the experimental report:
[Hyperparameter](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project__Optimization_of_hyperparameters_for_evolutionary_algorithms_.ipynb)
<details>
  <summary>Hyperparameter Evolutionary Algorithm Report</summary>
  


## Introduction

This technical report presents a comparative analysis of the performance of two distinct models trained for object detection tasks, targeting room, window, and door detection in images. Both models employed the YOLO framework. However, one model was subjected to hyperparameter optimization using an evolutionary algorithm under limited conditions, while the other utilized default settings.

## Model Specifications

- **Architecture:** 157 layers
- **Parameters:** 7,018,216
- **Gradients:** 0
- **GFLOPs:** 15.8

## Training Details

- **Dataset:** 2,148 images with 59,644 instances
- **Epochs for Hyperparameter Optimization:** 7 (recommended: 10)
- **Generations for Evolutionary Algorithm:** 30 (recommended: 300)
- **Training Epochs for Each Model:** 50

## Evaluation Metrics

Evaluated based on Precision (P), Recall (R), mAP50, and mAP50-95.

## Results

The following tables compare the performance of the models before and after hyperparameter optimization:

**Overall Performance:**

| Metric     | Optimized Model | Default Settings Model |
|------------|-----------------|------------------------|
| Precision  | 0.886           | 0.887                  |
| Recall     | 0.838           | 0.840                  |
| mAP50      | 0.905           | 0.904                  |
| mAP50-95   | 0.629           | 0.625                  |

**Performance by Class:**

| Class  | Metric     | Optimized Model | Default Settings Model |
|--------|------------|-----------------|------------------------|
| Room   | Precision  | 0.917           | 0.930                  |
|        | Recall     | 0.893           | 0.890                  |
|        | mAP50      | 0.946           | 0.946                  |
|        | mAP50-95   | 0.662           | 0.659                  |
| Window | Precision  | 0.864           | 0.859                  |
|        | Recall     | 0.806           | 0.807                  |
|        | mAP50      | 0.880           | 0.878                  |
|        | mAP50-95   | 0.502           | 0.495                  |
| Door   | Precision  | 0.875           | 0.871                  |
|        | Recall     | 0.814           | 0.822                  |
|        | mAP50      | 0.890           | 0.889                  |
|        | mAP50-95   | 0.723           | 0.721                  |

![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/52923e0d-5c62-4e6c-af6a-8e7ea2b261f2)


## Analysis

The minimal improvements observed in the optimized model could likely be attributed to the low number of iterations used during hyperparameter optimization: 7 epochs against a recommended 10, and 30 generations against a recommended 300. This limitation might have prevented the evolutionary algorithm from fully refining the model's hyperparameters.

Despite these limitations, the optimized model demonstrated slight improvements in overall mAP50-95 scores and recall for the 'room' class, indicating potential benefits of hyperparameter optimization, even with reduced iterations.

## Conclusion

Hyperparameter optimization through evolutionary algorithms can enhance YOLO-based object detection model performance, particularly in average precision across different IoU thresholds and specific class recalls. However, the extent of these improvements can be significantly influenced by the number of iterations and generations used in the optimization process. This analysis underscores the necessity of adequate optimization phases to achieve meaningful improvements in model performance.

This comparative study serves as a reference for developers and researchers focusing on the impact of hyperparameter tuning in deep learning models, emphasizing the need for sufficient optimization iterations to unleash the full potential of evolutionary algorithms.

</details>

### Learning Rate test
Adjusting evolutionary algorithm hyperparameters as the basis, I attempted the strategy mentioned in Isa et al. (2022) to reduce model oscillation by gradually decreasing the learning rate and improve model performance. I conducted a test with 50 epochs as a benchmark (as reducing the learning rate will slow down convergence) and increased momentum. This experiment was not fully completed, only testing the default SGD optimizer and a set of learning rates and momentums. Ideally, multiple combinations should be tested and attempt to use Adam optimizer instead of SGD as mentioned in Isa et al. (2022) study to complete this process. Below is the incomplete experimental result report:
[Learning Rate test](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/blob/main/Notebook/Deep_Learning_Project_learning_rate.ipynb)
<details>
  <summary>Learning Rate test Report</summary>

### Technical Report

#### Model Performance Comparison Before and After Adjustments

In the process of hyperparameter optimization for the YOLO v5 model, we compared the model performance before and after adjustments. Adjustments include changes in learning rate and momentum.

**Performance Comparison:**

| Metric     | Without LR Adjustment | With LR & Momentum Adjustment |
|------------|-----------------------|-------------------------------|
| **Overall (all)** P     | 0.886                  | 0.868                         |
| **Overall (all)** R     | 0.838                  | 0.830                         |
| **Overall (all)** mAP50 | 0.905                  | 0.900                         |
| **Overall (all)** mAP50-95 | 0.629                | 0.614                         |
| **Room** P              | 0.917                  | 0.891                         |
| **Room** R              | 0.893                  | 0.879                         |
| **Room** mAP50          | 0.946                  | 0.939                         |
| **Room** mAP50-95       | 0.662                  | 0.645                         |
| **Window** P            | 0.864                  | 0.850                         |
| **Window** R            | 0.806                  | 0.802                         |
| **Window** mAP50        | 0.880                  | 0.871                         |
| **Window** mAP50-95     | 0.502                  | 0.483                         |
| **Door** P              | 0.875                  | 0.862                         |
| **Door** R              | 0.814                  | 0.809                         |
| **Door** mAP50          | 0.890                  | 0.890                         |
| **Door** mAP50-95       | 0.723                  | 0.714                         |


![image](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/2f463453-2fad-4b0d-978d-03e1ddf4b708)

![ce1d6712-7a25-4214-b37e-092969413608](https://github.com/karin478/Artificial-neural-network-computer-vision-CAD-drawing-recognition/assets/79677735/4d11c586-7514-4dd5-86b4-282c2c6ffa71)


#### Hyperparameter Adjustments Explanation

- **Initial Learning Rate (lr0):** Adjusted from 0.01 to 0.01 (unchanged)
- **Final Learning Rate (lrf):** Reduced from 0.01 to 0.005
- **Momentum:** Increased from 0.973 to 0.999

The goal was to mitigate local optima by decreasing the learning rate and increasing momentum, allowing the model to fine-tune its weights more delicately over a greater number of training epochs.

#### Analysis of Results

After adjustments, performance across all categories showed a decrease, particularly in the mAP50 and mAP50-95 metrics. This suggests that:

1. **Learning Rate Adjustment:** Lowering the learning rate may have slowed the model's convergence, and within the given training epochs, the model might not have had sufficient time to reach an optimal state. Although theoretically, this method can reduce oscillations and help the model fit the data more finely, the actual effect might vary depending on the dataset characteristics and other model parameter configurations.

2. **Momentum Adjustment:** Increasing momentum helps the model move through the parameter space with a more steady velocity, aiding in escaping local optima. However, if momentum is set too high, it could cause the model to overly rely on the previous gradient direction, potentially ignoring new information from the data, which might explain why there was no performance improvement.
3. **Convergence trend:** An interesting observation from the graph is that although mAP50:95 did not surpass the original model due to the end of epochs, its upward trend indicates that it still has some distance to reach a plateau. This may suggest that decreasing the learning rate could enhance the model's ability for fine recognition. More experiments are needed to confirm this possibility.

#### Conclusion

While theoretically, a decreasing learning rate and increased momentum should aid model learning, these adjustments did not lead to performance improvements in practice but rather a decrease. This might indicate that the specific settings of these hyperparameters were not optimal for the current dataset and model configuration combination. In future work, considering more detailed parameter adjustment strategies and longer training periods may allow the model more time to adapt to the data. Additionally, exploring different optimizers and learning rate scheduling strategies could help find the most suitable configuration for the current task.


</details>

### The impact of batch size selection during validation on prediction performance.(Unfinished)
After the experiment ended, I found that the validation results obtained by individually calling the val.py function for validation of the validation set were not consistent with the automatic validation set detection results after training. Due to time constraints in this experiment, I uniformly chose to use the training report as a horizontal comparison of results (except for prototypes with many training errors in the early stage), but I believe that such a comparative result is not rigorous enough. If we stay at an academic level, we can simply list out the observation results. However, as part of an enterprise solution, my task is to continuously improve model prediction performance. Therefore, I have started batch size testing in the model verification phase.

<details>
  <summary>Validation batch size test Report</summary>

### Methodology
 We selected a previously saved model. The model had good validation performance on the training summary's validation set. We will import the model and test its predictive performance with batch sizes of 32, 16, 8, 4, and 2.

### Result

**Performance Comparison:**

| Batch | Class  | Images | Instances |     P |     R | mAP50 | mAP50-95 |
|-------|--------|--------|-----------|-------|-------|-------|----------|
| 32    | all    |   2148 |     60528 | 0.907 | 0.829 | 0.897 |    0.634 |
| 32    | room   |   2148 |     22640 | 0.927 | 0.879 | 0.924 |    0.656 |
| 32    | window |   2148 |     14820 |  0.88 | 0.789 | 0.868 |     0.51 |
| 32    | door   |   2148 |     23068 | 0.912 |  0.82 | 0.899 |    0.735 |
| 16    | all    |   2148 |     60528 | 0.907 | 0.844 | 0.911 |    0.645 |
| 16    | room   |   2148 |     22640 | 0.929 | 0.896 | 0.941 |     0.67 |
| 16    | window |   2148 |     14820 | 0.879 | 0.801 | 0.878 |    0.517 |
| 16    | door   |   2148 |     23068 | 0.914 | 0.837 | 0.916 |    0.748 |
| 8     | all    |   2148 |     60528 | 0.907 | 0.856 | 0.924 |    0.652 |
| 8     | room   |   2148 |     22640 | 0.928 | 0.908 | 0.953 |    0.677 |
| 8     | window |   2148 |     14820 |  0.88 | 0.813 | 0.891 |    0.524 |
| 8     | door   |   2148 |     23068 | 0.914 | 0.847 | 0.928 |    0.757 |
| 4     | all    |   2148 |     60528 | 0.907 | 0.859 | 0.927 |    0.655 |
| 4     | room   |   2148 |     22640 | 0.929 | 0.912 | 0.957 |    0.679 |
| 4     | window |   2148 |     14820 |  0.88 | 0.815 | 0.895 |    0.525 |
| 4     | door   |   2148 |     23068 | 0.914 |  0.85 |  0.93 |    0.759 |
| 2     | all    |   2148 |     60528 | 0.907 |  0.86 | 0.928 |    0.655 |
| 2     | room   |   2148 |     22640 | 0.929 | 0.913 | 0.958 |     0.68 |
| 2     | window |   2148 |     14820 | 0.879 | 0.816 | 0.896 |    0.526 |
| 2     | door   |   2148 |     23068 | 0.914 |  0.85 | 0.931 |     0.76 |

## Overall Trends:

- **Precision (P)**: Remains constant across all batches at 0.907. This indicates the model's ability to correctly identify positive instances does not significantly change with batch size.
- **Recall (R)**: Shows a slight increasing trend as the batch size decreases, moving from 0.829 in batch 32 to 0.860 in batch 2. This suggests smaller batch sizes may slightly improve the model's ability to find all positive instances.
- **mAP50**: Increases slightly as the batch size decreases, from 0.897 in batch 32 to 0.928 in batch 2, indicating a slight improvement in the accuracy of object detection.
- **mAP50-95**: Also shows a slight increase as batch size decreases, suggesting an improvement in detection across varying object sizes and conditions.

## Room Class:

- **Precision and Recall**: Both improve as the batch size decreases, indicating better identification and retrieval of 'room' instances with smaller batches.
- **mAP50 and mAP50-95**: Increase with decreasing batch size, suggesting overall improvements in detection accuracy and robustness for the 'room' class with smaller batches.

## Window Class:

- **Precision**: Remains fairly constant, slightly fluctuating around 0.88. This suggests the model's precision for detecting windows is not significantly affected by batch size.
- **Recall**: Improves as the batch size decreases, from 0.789 to 0.816, indicating that smaller batches might help in retrieving more true positive window instances.
- **mAP50 and mAP50-95**: Improve as the batch size decreases, although the changes are not as significant as those for the 'room' class.

## Door Class:

- **Precision**: Slightly increased in smaller batches, moving from 0.912 in batch 32 to 0.914 in batches 8, 4, and 2.
- **Recall**: Increases more noticeably from 0.82 in batch 32 to 0.85 in batch 2, indicating improved ability to find all relevant instances of doors with smaller batches.
- **mAP50 and mAP50-95**: Show the most significant improvements in the 'door' class as batch size decreases, suggesting this class benefits particularly from smaller batch sizes in terms of detection accuracy and robustness.

## Discussion:



</details>



## Limitations of the experiment
1. In this experiment, the vast majority of parameters and model enhancement tests used fixed epoch and image parameters. A more standardized experimental approach is to first conduct a large number of epochs, such as 100 or even 200. Then observe the model's loss reduction and mAP convergence as the number of epochs increases. Finally, select the appropriate epoch to avoid underfitting and overfitting issues. However, due to the short duration of this experiment, a simpler fixed parameter method was chosen. In the early stages of the experiment, a fixed 50 epochs were used as a baseline. It was observed during the experiment that the model would generally converge between 20-30 epochs, so in later stages of the experiment it was changed to 30 epochs. This is just a quick application method and not rigorous or precise enough for standard practice.
2. During the training process of the model, a Colab notebook was used as the experimental environment for demonstration purposes. The main goal is to showcase the process of improving the model experiment, so not all models and related files from each experimental period were saved (only saving the best result model as output). However, for more formal projects with longer time spans and multiple collaborators, all model files, configuration files, and related data should be saved and annotated.
3. In this experiment, although the performance of the YOLO model was tested. But because the application scenario was not determined from the beginning, it is difficult to set a threshold. For example, if it is for real-time detection, then a higher FPS will be set as a threshold for experimental benchmarking and exclude models with high accuracy but slow detection speed. For example, whether the model is deployed on a server and then remotely connected to terminal devices to send data or run directly on small terminal devices? This will affect the choice of YOLO model size. If you can determine the purpose of an experiment from the beginning, it will better guide the experimental process.
4. This experiment is just a simple demonstration and research. It has more academic nature and lacks practical application. For example, we only tested the performance of the model in the training set and validation machine. However, if we enter the industrial field, we will need to convert the notebook into a complex architecture, including building data pipelines, regular updates of models, automatic testing of model outputs, handling errors, new issues when training on large-scale data (longer time span, parallel CPU computing, estimation of computational resource requirements), and a complete set of standardized documentation.
5. For this experiment, in order to better demonstrate the purpose, a colab notebook was chosen. However, this brings up an issue that there is a maximum limit on the running time of each session. This prevents us from conducting more experiments in one notebook. This may not seem like a big problem at first glance. But when it comes to horizontal comparison of data, besides the baseline of each notebook (such as the former being compared), comparisons with other notebooks will be weakened. Due to the randomness of sample splitting and neural network initialization, even if we specify the same seed parameter, it is difficult to reproduce past experimental results.
6. In this experiment, the evaluation method of using val.py to validate the results of the dataset still needs further research. We encountered different situations when validating the dataset with val.py separately at the beginning and end of the experiment compared to the training summary report. The early val.py results were much lower than the training summary, while in later stages, after selecting a smaller batch size, the results exceeded those of the training summary instead. Despite conducting data comparisons, I do not believe this is a research direction that can be ignored. If considering academic research, all batch sizes should be thoroughly tested for their impact on the validation process. Since we have not had a chance to fully test how different batch sizes affect model training during testing yet, this comprehensive testing may need to be done after testing various batch sizes during training. Considering enterprise application scenarios, we should use smaller batch sizes to improve predictive performance only when multiple tests show stable validation results.
7. In this experiment, all data tests were conducted only once. Due to the inherent uncertainty of neural networks, it is preferable to conduct multiple tests to obtain an average value or use cross-validation methods (given that the dataset contains just over 2000 images, cross-validation may be a suitable approach).


## Other related research and thinking

### Engineering drawing recognition using OCR and YOLO
[Integration of Deep Learning for Automatic Recognition of 2D Engineering Drawings](https://www.mdpi.com/2075-1702/11/8/802)
In this study, researchers first used YOLO training to enable it to distinguish different graphic features and symbols in the drawings from text features. Through YOLO, text is found and extracted for OCR text extraction. The extracted results are then combined with the partially recognized images from YOLO, achieving a dual recognition effect of both text and images. Users can simultaneously obtain image and text information. This is a very practical construction method that is not difficult to implement. If there is time, we will attempt this approach.

- Lin, Y.-H.; Ting, Y.-H.; Huang, Y.-C.; Cheng, K.-L.; Jong, W.-R. Integration of Deep Learning for Automatic Recognition of 2D Engineering Drawings. Machines 2023, 11, 802. https://doi.org/10.3390/machines11080802

### Optimizing the Hyperparameter Tuning of YOLOv5 for Underwater Detection
[Optimizing the Hyperparameter Tuning of YOLOv5 for Underwater Detection](https://ieeexplore.ieee.org/document/9773108)
In this study, the authors compared the impact of different versions of YOLO (3-5) on detecting underwater objects. The study highlights that YOLO has strong image recognition capabilities, especially in YOLO v5 version. Another focus of this research is hyperparameter tuning. Researchers used a model optimization technique with adaptive learning rates and concluded that the combination of hyperparameter optimization and adaptive learning rates can further enhance the model performance of YOLO v5.

- Isa, I. S., Rosli, M. S. A., Yusof, U. K., Maruzuki, M. I. F., & Sulaiman, S. N. (2022). Optimizing the hyperparameter tuning of YOLOv5 for underwater detection. IEEE Access, 10, 52818-52831.

### Automatic License Plate Recognition via sliding-window darknet-YOLO deep learning
[Automatic License Plate Recognition via sliding-window darknet-YOLO deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0262885619300575?via%3Dihub)
This article uses the YOLO model to detect vehicle license plates. One valuable aspect of this study is the use of two layers of YOLO for detection. The first layer is used to extract license plates from photos. The second step involves a secondary recognition by YOLO of the specific content on the license plate. By establishing targeted data pipelines and fine-tuning YOLO before images and between the two types of images (photos and license plates), better results can be achieved. In this project's drawings, since they contain symbols and text, this method can also be applied effectively. By building a multi-layered YOLO image recognition model for individual fine-tuning and data pipeline construction, it should further enhance the ability to recognize architectural drawings.

- Chen, R. C. (2019). Automatic License Plate Recognition via sliding-window darknet-YOLO deep learning. Image and Vision Computing, 87, 47-56.

### Overview of YOLO Object Detection Algorithm
[Overview of YOLO Object Detection Algorithm](https://doi.org/10.56028/ijcit.1.2.11)
This study compares the algorithm upgrades of different versions of YOLO, as well as the advantages and disadvantages of YOLO. From this article, it is understood that through the iterations of YOLO versions, YOLO has introduced techniques such as batch normalization and feature extraction for iteration. Most importantly, through this article I learned about the differences and pros and cons between the architecture of YOLO and the traditional two-stage detection R-CNN architecture. Traditional R-CNN usually requires two calculations: first, an algorithm divides a whole image into multiple small candidate regions; secondly, a CNN model is used to perform convolution operations on these candidate regions to obtain results. On the other hand, YOLO divides the entire image into numerous grids where each grid calculates whether it contains an object in one single calculation for the entire image. The advantage of YOLO's algorithm is its faster speed, allowing real-time image recognition (such as applications in cameras and VR technology), while its disadvantage is that it may have lower accuracy in detecting very small objects compared to traditional R-CNN. Of course, through continuous iterations and different topics, this conclusion could also change. Or just like what I learned from previous studies - by applying YOLO multiple times (first for position determination and category recognition, second time for detailed identification or OCR), this problem can be effectively solved.

- Wan, C., Pang, Y., & Lan, S. (2022). Overview of YOLO Object Detection Algorithm. International Journal of Computing and Information Technology, 2(1), 11-11.

### In object detection deep learning methods, YOLO  shows supremum to Mask R-CNN
[In object detection deep learning methods, YOLO  shows supremum to Mask R-CNN](https://iopscience.iop.org/article/10.1088/1742-6596/1529/4/042086)
In this study, the authors compared the R-CNN network traditionally considered to have higher precision with the YOLO network. The objects of comparison included various variants of R-CNN, such as Mask R-CNN and Fast R-CNN. Through comparison, researchers believe that Mask R-CNN can perform tasks under unlabeled conditions compared to YOLO, which YOLO cannot achieve. Furthermore, Mask R-CNN can continuously improve model accuracy through a large amount of data. This indicates that Mask R-CNN requires a large quantity of data but lower quality requirements than YOLO. In contrast, after training completion, YOLO can detect various types of situations, and its detection breadth and speed are considered superior to Mask R-CNN.

- Sumit, S. S., Watada, J., Roy, A., & Rambli, D. R. A. (2020, April). In object detection deep learning methods, YOLO shows supremum to Mask R-CNN. In Journal of Physics: Conference Series (Vol. 1529, No. 4, p. 042086). IOP Publishing.

### Yolo V4 for Advanced Traffic Sign Recognition With Synthetic Training Data Generated by Various GAN
[Yolo V4 for Advanced Traffic Sign Recognition With Synthetic Training Data Generated by Various GAN](https://ieeexplore.ieee.org/document/9471877)
In this study, researchers used GAN (Generative Adversarial Network) to generate images (DCGAN, LSGAN, and WGAN), and then mixed the images into the training set for YOLO v4 model training. The results showed that this approach significantly improved the model's recognition ability.

- Dewi, C., Chen, R. C., Liu, Y. T., Jiang, X., & Hartomo, K. D. (2021). Yolo V4 for advanced traffic sign recognition with synthetic training data generated by various GAN. IEEE Access, 9, 97228-97242.


### Future development ideas
#### Combine ORC technology.
Just like in the license plate research. OCR technology can be fully utilized to enhance the recognition ability of YOLO model. Break through the limitations of the model. Through OCR, we can better rotate and extract information from paper fragments, combine the extracted text information with image information. By combining multiple layers of YOLO and OCR, a practical enhancement that introduces textual information as one dimension into overall applications can be constructed.

#### For different types of tasks, select the YOLO model for two rounds of construction
The YOLO model can be stacked in multiple layers. Such applications have been seen in past research. By using different types of data preprocessing and YOLO training (pre-model responsible for image segmentation, post-model responsible for detecting small objects), a two-step calculation model similar to the traditional R-CNN is formed. However, benefiting from the speed and continuous iteration of the YOLO model, this solution may surpass the performance limitations of existing YOLO models while also exceeding the accuracy of traditional two-step models.

#### Use a more comprehensive evolutionary algorithm
When more computing resources and time are available, increase the number of epochs and evolutions. Training in parallel by concatenating multiple GPUs can maximize the performance of the model.

#### More refined hyperparameter tuning
According to past research, good model performance can also be achieved by manually adjusting parameters. Accumulate experience and try many interesting hyperparameter settings through more paper studies. For example, the best model performance was obtained by using SGD+momentum adjustment+gradual decrease in learning rate as used in previous research.

### Improve recognition performance by building adversarial models.
By constructing an adversarial model, generating visually indistinguishable erroneous images for training. It can effectively enhance the robustness and generalization performance of the model.

#### Try other models
There are many models to try and compare, such as SSD, Mask R-CNN, Fast R-CNN, etc. The best model may vary depending on the application scenario. For example, real-time monitoring cameras and VR require one-shot calculation models like SSD and YOLO; while for retrieving from databases and building large language model robots, you may choose the R-CNN series models with possibly higher accuracy but lower efficiency.



