# 西电瞎几把搞 心电人机智能大赛

## 数据集

- 将代码clone到本地
- 代码默认是调用项目里`.data/`目录下的文件，因此需要新建目录`.data/`
- 下载数据集并解压
  - arrythmia
    http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231754/round1/hf_round1_arrythmia.txt
  - 训练集
    - train
      http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231754/round1/hf_round1_train.zip
    - label
      http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231754/round1/hf_round1_label.txt
  - 测试集
    - testA
      http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231754/round1/hf_round1_testA.zip
    - subA
      http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231754/round1/hf_round1_subA.txt
- 解压后应有`data/testA`,`data/train`两个目录

## 训练

目前参数没有调，模型瞎几把建的，但是在验证集上已经有86%的f1得分

## 生成提交数据

9/17晚上提交的显示格式错误报错，我改了代码重新提交，明天早上看结果

## TODO

- 实验不同的模型
  - 我这是用1D卷积做的，参考了一下resnet的结构，但是借鉴的并不完美
  - 接下来可以尝试
    - 完美移植一波resnet(两种不同block结构)
    - 实验RNN效果（个人直觉没软用）
    - 实验其他网络结构
- 调整模型参数，寻找最优解