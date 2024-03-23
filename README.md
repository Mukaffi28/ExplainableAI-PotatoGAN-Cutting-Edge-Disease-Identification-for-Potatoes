# PotatoGANs: Utilizing Generative Adversarial Networks, Instance Segmentation, and Explainable AI for Enhanced Potato Disease Identification and Classification
## Abstract
Numerous applications have resulted from the automation of agricultural disease segmentation using deep learning techniques. However, when applied to new conditions, these applications frequently face the difficulty of overfitting, resulting in lower segmentation performance. In the context of potato farming, where diseases have a large influence on yields, it is critical for the agricultural economy to quickly and properly identify these diseases. Traditional data augmentation approaches, such as rotation, flip, and translation, have limitations and frequently fail to provide strong generalization results. To address these issues, our research employs a novel approach termed as PotatoGANs. In this novel data augmentation approach, two types of Generative Adversarial Networks (GANs) are utilized to generate synthetic potato disease images from healthy potato images. This approach not only expands the dataset but also adds variety, which helps to enhance model generalization. Using the Inception score as a measure, our experiments show the better quality and realisticness of the images created by PotatoGANs, emphasizing their capacity to closely resemble real disease images. The CycleGAN model outperforms the Pix2Pix GAN model in terms of image quality, as evidenced by its higher IS scores CycleGAN achieves higher Inception scores (IS) of 1.2001 and 1.0900 for black scurf and common scab, respectively. This synthetic data can significantly improve the training of large neural networks. It also reduces data collection costs while enhancing data diversity and generalization capabilities. Our work improves interpretability by combining three gradient-based Explainable AI algorithms (GradCAM, GradCAM++, and ScoreCAM) with three distinct CNN architectures (DenseNet169, Resnet152 V2, InceptionResNet V2) for potato disease classification. This comprehensive technique improves interpretability with insightful visualizations and provides detailed insights into the network's decision-making. The goal of combining several CNN designs and explanation techniques is to maximize interpretability while offering an in-depth understanding of the model's behavior. We further employ this extended dataset in conjunction with Detectron2 to segment two classes of potato disease images, with the primary goal of enhancing the overall performance of our model. Furthermore, in the ResNeXt-101 backbone, Detectron2 has a maximum dice score of 0.8112. This combination of PotatoGANs and Detectron2 has the potential to be a powerful approach for tackling the limitations of traditional data augmentation approaches while also boosting the accuracy and robustness of our disease segmentation model.

## Table of Contents
- [Experimental Setups](#experimental-setups)
- [Dataset Availability](#dataset-availability)
- [Results](#results)
- [Citation](#citation)
- [Contact Information](#contact-information)

## Experimental Setups

### Setup 1: Kaggle
- **Environment:**
  - Python Version: 3.11
  - PyTorch Version: 2.1.0
  - GPU: T4 GPU with 7.5 Compute Capability
  - RAM: 30 GB

### Setup 2: Jupyter Notebook Environment
- **Environment:**
  - Python Version: 3.10.12
  - PyTorch Version: 2.1.0
  - GPU: NVIDIA GeForce RTX 3050 (8 GB)
  - RAM: 16 GB
  - Storage: 512 GB NVMe SSD

### Setup 2: Jupyter Notebook Environment
- **Environment:**
  - Python Version: 3.10.12
  - Tensforflow Version: 2.6.0
  - GPU: NVIDIA GeForce RTX 3050 (8 GB)
  - RAM: 16 GB
  - Storage: 512 GB NVMe SSD
    
## Dataset Availability

The Comprehensive Potato Disease Dataset is now publicly accessible! This dataset, available in both jpg and png formats, offers a valuable resource for diverse research and analysis purposes. You can explore and download the dataset at the following link: [Dataset](https://github.com/Wasi34/Comprehensive-Potato-Disease-Dataset). Feel free to utilize this resource for your research, experiments, or any analytical endeavors. Should you have any questions or require further assistance with the dataset, please don't hesitate to reach out.


## Results
### Generated Potato Disease Realistic Image Evaluation Using Frechet Inception Distance and Inception Score

| **Class**      | **GANs**       | **Frechet Inception Distance** | **Inception Score** |
|----------------|----------------|--------------------------------|---------------------|
| Black Scurf    | Cycle GAN      | 0.4028                         | 1.2001              |
|                | Pix2Pix GAN    | 0.5743                         | 0.9899              |
| Common Scab    | Cycle GAN      | 0.4882                         | 1.0900              |
|                | Pix2Pix GAN    | 0.6240                         | 0.9643              |


### Performance Evaluation of Pretrained CNN for Potato Disease Classification

| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Log Loss** |
|----------------------|--------------|---------------|------------|--------------|--------------|
| DenseNet169         | 1.0000       | 1.0000        | 1.0000     | 1.0000       | 0.0024       |
| Resnet152V2         | 0.9804       | 0.9792        | 0.9821     | 0.9803       | 0.7067       |
| InceptionResNetV2   | 0.9902       | 0.9912        | 0.9891     | 0.9901       | 0.3533       |


### Performance Evaluation of Potato Disease Instance Segmentation

| **Backbone** | **Task Type** | **AP** | **\(AP_{IoU= 0.5}\)** | **\(AP_{IoU= 0.75}\)** | **Dice Score** |
|--------------|---------------|--------|-------------------------|-------------------------|----------------|
| ResNet-50    | Segmentation  | 73.204 | 89.733                  | 86.126                  | 0.6014         |
|              | Bounding Box  | 83.824 | 90.526                  | 86.353                  |                |
|--------------|---------------|--------|-------------------------|-------------------------|----------------|
| ResNet-101   | Segmentation  | 78.681 | 92.905                  | 74.851                  | 0.6728         |
|              | Bounding Box  | 87.886 | 96.409                  | 90.943                  |                |
|--------------|---------------|--------|-------------------------|-------------------------|----------------|
| ResNeXt-101  | Segmentation  | 86.039 | 97.030                  | 96.040                  | 0.8112         |
|              | Bounding Box  | 97.030 | 97.030                  | 97.030                  |                |


## Contact Information

For any questions, collaboration opportunities, or further inquiries, please feel free to reach out:

- **Fatema Tuj Johora Faria**
  - Email: [fatema.faria142@gmail.com](mailto:fatema.faria142@gmail.com)

- **Mukaffi Bin Moin**
  - Email: [mukaffi28@gmail.com](mailto:mukaffi28@gmail.com)

- **Ahmed Al Wase**
  - Email: [ahmed.alwasi34@gmail.com](mailto:ahmed.alwasi34@gmail.com)
    
## Citation

If you find the "Vashantor" dataset or the associated research work helpful, please consider citing our paper:

```bibtex
@misc{faria2023vashantor,
  title={Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language},
  author={Fatema Tuj Johora Faria and Mukaffi Bin Moin and Ahmed Al Wase and Mehidi Ahmmed and Md. Rabius Sani and Tashreef Muhammad},
  year={2023},
  eprint={2311.11142},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

