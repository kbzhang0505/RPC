# Be An Excellent Student: Review, Preview, and Correction
We propose a novel yet effective knowledge distillation scheme which mimics an all-round learning process of an excellent student from the teacher, i.e., knowledge preview, knowledge review, and knowledge correction, to acquire more informative and complementary knowledge for performance improvement. In the newly proposed method, to better leverage more comprehensive feature knowledge for the teacher model, we propose Knowledge Review and Knowledge Preview Distillation to amalgamate multi-level features from different intermediate layers in both forward and backward pathways and fully distill them through hierarchical context loss, which greatly improves the student’s feature learning efficiency. Moreover, we further present a Response Correction Mechanism to reinforce the prediction of student, which can more fully excavate the student’ s own knowledge, effectively alleviating the negative influence caused by the knowledge gap between the teacher and the student. We verify the effectiveness of our method with various networks on the CIFAR-100 datasets and the proposed method achieves competitive results compared with other state-of-the-art competitors.
# The overall framework
![Overall](https://github.com/kbzhang0505/RPC/assets/97494153/f5ef7938-e72b-4b6f-899e-1bf184ad4d98)
# Knowledge Review and Preview Distillation
![Fig3](https://github.com/kbzhang0505/RPC/assets/97494153/b8c3019f-ee50-4ee0-9020-687832e2171a)
# Response Correction Mechanism
![Fig5](https://github.com/kbzhang0505/RPC/assets/97494153/21fc38c7-4c07-484f-aab1-e49aed2c5c93)
### Dataset Structure ###
//For training, you need to build the new directory.

*├─data 

**└─Cifar100 

*├─logs 
### Weights ###
The weights of student models are available at https://pan.baidu.com/s/1d9CkeMjBfWEU3DPPXR8Pjg?pwd=0207.
### Citation ###
If you find this code and data useful, please consider citing citing our paper:
```
@ARTICLE{10319075,
  author={Cao, Qizhi and Zhang, Kaibing and He, Xin and Shen, Junge},
  journal={IEEE Signal Processing Letters}, 
  title={Be An Excellent Student: Review, Preview, and Correction}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LSP.2023.3333240}}
```
The code is based on ReviewKD-master.
### Thanks ###
```
@INPROCEEDINGS{9578915,
  author={Chen, Pengguang and Liu, Shu and Zhao, Hengshuang and Jia, Jiaya},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Distilling Knowledge via Knowledge Review}, 
  year={2021},
  volume={},
  number={},
  pages={5006-5015},
  doi={10.1109/CVPR46437.2021.00497}}
```
# If there are any questions，please feel free to contact us.
