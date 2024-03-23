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

The "Potato-Disease" dataset, available in both CSV and JSON formats, is now publicly accessible. This dataset provides users with a valuable opportunity for flexible exploration and utilization in various research and analysis endeavors. You can explore and download the dataset at the following link: [Vashantor Dataset](https://data.mendeley.com/datasets/bj5jgk878b/2) Feel free to leverage this resource for your research, experiments, or any other analytical purposes. If you have any questions or need further assistance with the dataset, don't hesitate to reach out.

## Results
### CER, WER, BLEU, METEOR scores of all the Bangla regional dialect translation models

| Region       | Model    | CER    | WER    | BLEU   | METEOR |
|--------------|----------|--------|--------|--------|--------|
| Chittagong   | mT5      | 0.2308 | 0.3959 | 36.75  | 0.6008 |
| Chittagong   | BanglaT5 | 0.2040 | 0.3385 | 44.03  | 0.6589 |
| Noakhali     | mT5      | 0.2035 | 0.3870 | 37.43  | 0.6073 |
| Noakhali     | BanglaT5 | 0.1863 | 0.3214 | 47.38  | 0.6802 |
| Sylhet       | mT5      | 0.1472 | 0.2695 | 51.32  | 0.7089 |
| Sylhet       | BanglaT5 | 0.1715 | 0.2802 | 51.08  | 0.7073 |
| Barishal     | mT5      | 0.1480 | 0.2644 | 48.56  | 0.7175 |
| Barishal     | BanglaT5 | 0.1497 | 0.2459 | 53.50  | 0.7334 |
| Mymensingh   | mT5      | 0.0796 | 0.1674 | 64.74  | 0.8201 |
| Mymensingh   | BanglaT5 | 0.0823 | 0.1548 | 69.06  | 0.8312 |

These results represent the evaluation metrics (CER, WER, BLEU, METEOR) for different regions using the mT5 and BanglaT5 models. Explore the performance of each model across various linguistic regions.

### Performance Overview of all region detection models

| Model              | Accuracy | Log Loss | Region       | Precision | Recall  | F1-Score |
|---------------------|----------|----------|--------------|-----------|---------|----------|
| mBERT              | 84.36%   | 0.9549   | Chittagong   | 0.8779    | 0.8058  | 0.8913   |
|                    |          |          | Noakhali     | 0.9286    | 0.9437  | 0.9361   |
|                    |          |          | Sylhet       | 0.7304    | 0.9013  | 0.8072   |
|                    |          |          | Barishal     | 0.8187    | 0.5893  | 0.6847   |
|                    |          |          | Mymensingh   | 0.9412    | 0.968   | 0.9544   |
| Bangla-bert-base   | 85.86%   | 0.8804   | Chittagong   | 0.884     | 0.8486  | 0.8651   |
|                    |          |          | Noakhali     | 0.9625    | 0.9301  | 0.9461   |
|                    |          |          | Sylhet       | 0.7388    | 0.9147  | 0.8173   |
|                    |          |          | Barishal     | 0.8373    | 0.616   | 0.7091   |
|                    |          |          | Mymensingh   | 0.9599    | 0.9653  | 0.9626   |

These metrics showcase the performance of mBERT and Bangla-bert-base models in terms of accuracy, log loss, precision, recall, and F1-score across different regions.

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

