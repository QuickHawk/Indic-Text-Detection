# Indic Text Detection

This project is part of Project Phase - I of Graduate Program in Visvesvaraya National Institute of Technology, Nagpur.

The project consists of files realted to training and evaluation of model. The model is available in [HuggingFace](https://huggingface.co/QuickHawk/indic-text-detection)

## Descriptions

This model extends `openmmlab/upernet-swin-base` and performs binarization which generates the segmentation map.

![Proposed_Model.jpg](https://cdn-uploads.huggingface.co/production/uploads/6868f8219c4cd7445653ada1/d0hK3K7xPY3MfIr_0ynB0.jpeg)

## Example

![Detection_Anamoly_2.jpg](https://cdn-uploads.huggingface.co/production/uploads/6868f8219c4cd7445653ada1/XgtugD9mHrWnggbGRJjbv.jpeg)

## Evaluation Metrics

| **Metric**  | **Tiny** | **Base** |
|-------------|----------|----------|
| Precision   | 0.8352   | 0.8628   |
| Recall      | 0.8411   | 0.8521   |
| F-Score     | 0.8381   | 0.8574   |

