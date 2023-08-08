# RPC
We propose a novel yet effective knowledge distillation scheme which mimics an all-round learning process of an excellent student from the teacher, i.e., knowledge preview, knowledge review, and knowledge correction, to acquire more informative and complementary knowledge for performance improvement. In the newly proposed method, to better leverage more comprehensive feature knowledge for the teacher model, we propose Knowledge Review and Knowledge Preview Distillation to amalgamate multi-level features from different intermediate layers in both forward and backward pathways and fully distill them through hierarchical context loss, which greatly improves the student’s feature learning efficiency. Moreover, we further present a Response Correction Mechanism to reinforce the prediction of student, which can more fully excavate the student’ s own knowledge, effectively alleviating the negative influence caused by the knowledge gap between the teacher and the student. We verify the effectiveness of our method with various networks on the CIFAR-100 datasets and the proposed method achieves competitive results compared with other state-of-the-art competitors.
# The overall framework
![Overall](https://github.com/kbzhang0505/RPC/assets/97494153/f5ef7938-e72b-4b6f-899e-1bf184ad4d98)
# Knowledge Review and Preview Distillation
![Fig3](https://github.com/kbzhang0505/RPC/assets/97494153/b8c3019f-ee50-4ee0-9020-687832e2171a)
# Response Correction Mechanism
![Fig5](https://github.com/kbzhang0505/RPC/assets/97494153/21fc38c7-4c07-484f-aab1-e49aed2c5c93)
# Comparisons with State-of-the-Art Methods
![table](https://github.com/kbzhang0505/RPC/assets/97494153/ef217a55-2ad5-4bf0-9c75-8d69202f0281)
# Visualization comparisons of feature maps and correlation matrices
![visual](https://github.com/kbzhang0505/RPC/assets/97494153/f97bdf72-026d-43c3-ba6b-b38138dfc8e9)
# Robustness Evaluation
![Fig4](https://github.com/kbzhang0505/RPC/assets/97494153/d63d8c2a-3098-4b77-9ea8-e6275070ace0)

