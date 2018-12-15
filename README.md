
该模型是一个基于字符的BiLSTM-CRF序列标注模型。

运行代码环境：Python 3 和TensorFlow 1.2


### 模型介绍

整个模型共分三层:

第一层：向量查找层。目的是将输入的字符汉字转化为对应的字符向量（采用的是one-hot方法）

第二层：双向LSTM。目的是有效地自动提取输入信息的特征。

第三层：CRF层。顶层使用CRF对句子中的字符打标签，完成标注。

![Network](./pic/network.png)

### 训练方法
输入如下命令，开始训练模型

`python main.py --mode train --dataset_name MSRA`

语料库选择，修改`--dataset_name`参数（MSRA, ResumeNER, WeiboNER,人民日报）

使用预训练的字向量，设置参数`--use_pre_emb true`，默认为false

备注：(增加了自动选择对应数据集tag的功能)

~~训练其他语料库的话，由于不同语料库的**实体类别**可能存在差异，需要修改`data.py`代码中的tag2label~~，
如果需要运行demo，还需要修改`utils.py`里的`get_entity()`系列方法

| MSRA实体类别 | 标签(BIO标记法) |
| ------ | ------ |
| 人名  | B-PER I-PER |
| 地名  | B-LOC I-LOC |
| 机构名 | B-ORG I-ORG|

| 人民日报实体类别 | 标签(BIO标记法) |
| ------ | ------ |
| 人名       | B-PERSON I-PERSON |
| 普通地名    | B-LOC I-LOC |
| 行政区划地名 | B-GPE I-GPE |
| 机构名 | B-ORG I-ORG|
| 其他   | B-MISC I-MISC|

| WeiboNER实体类别 | 标签(BIO标记法) |
| ------ | ------ |
| 人名  | B-PER.NAM I-PER.NAM |
| 地名  | B-LOC.NAM I-LOC.NAM |
| 机构名 | B-ORG.NAM I-ORG.NAM|
| GPE-political | B-GPE.NAM I-GPE.NAM|
| 人名(nominal)  | B-PER.NOM I-PER.NOM |
| 地名(nominal)  | B-LOC.NOM I-LOC.NOM |
| 机构名(nominal) | B-ORG.NOM I-ORG.NOM|

| ResumeNER实体类别 | 标签(IOBES标记法) |
| ------ | ------ |
| 人名  | B-NAME M-NAME  E-NAME、S-NAME |
| 民族/种族  | B-RACE M-RACE  E-RACE、S-RACE |
| 国家  | B-CONT M-CONT E-CONT、S-CONT  |
| 地名 | B-LOC M-LOC E-LOC、S-LOC|
| 专业 | B-PRO M-PRO E-PRO、S-PRO|
| 学历 | B-EDU M-EDU E-EDU、S-EDU|
| 职位 | B-TITLE M-TITLE E-TITLE、E-TITLE|
| 组织机构 |  B-ORG M-ORG E-ORG 、S-ORG|



### 测试方法
输入如下命令完成测试集测试

`python main.py --mode test --dataset_name MSRA --demo_model 1522858865`

备注:训练过程中，每开始一次都会在“data_path_save/”目录下产生一个文件夹(以时间转换为整数来命名的)，将训练的参数保存。
     当测试的时候，想用哪次训练的参数进行测试，就将该次训练的文件名赋值给“--demo_model"，即替换上面命令中的"1522858865"。
     ”1522858865“是我在训练时的最后参数。



### 演示
在这里可以输入一段文本，查看识别结果。

运行命令如下;

`python main.py --mode demo --dataset_name MSRA --demo_model 1522858865`

运行程序后，会提示输入一段文本，输入后就可以看到通过该代码识别的结果。

![demo](./pic/demo.PNG)

### 参考
\[1\] [Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF)