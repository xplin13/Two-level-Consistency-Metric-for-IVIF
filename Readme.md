# *Two-level Consistency Metric for Infrared and Visible Image Fusion*

## Platform
python 3.7.0\
torch 1.4.0\
torchvision 0.5.0

## Fusion framework
MSSD is utilized to train our network. The dataset can be downloaded at https://drive.google.com/drive/folders/1QE3xytmOJ-zNGUobJyFP_J0JatxszjR3
![图片](./demonstrate_images/network.png)

Our trained model can be downloaded at: https://pan.baidu.com/s/14wEtKrzXypk8ReVg6meXLg?pwd=8jny .\
The model should be put into checkpoint folder for test.

## Update
In a subsequent study, we found that combine the two types of high-frequency attention source images in the computation of the weight map could better preserve the  saliency of the infrared target in the fusion results and obtain better visual quality. Therefore, we update the loss function named wse_update here, and if you use this loss function, please cite the article:\

@article{lin2022two,\
  title={Two-Level Consistency Metric for Infrared and Visible Image Fusion},\
  author={Lin, Xiaopeng and Zhou, Guanxing and Tu, Xiaotong and Huang, Yue and Ding, Xinghao},\
  journal={IEEE Transactions on Instrumentation and Measurement},\
  volume={71},\
  pages={1--13},\
  year={2022},\
  publisher={IEEE}\
}


