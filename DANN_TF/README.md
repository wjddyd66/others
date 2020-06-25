### Project
GitHub: https://github.com/wjddyd66/others/tree/master/DANN_TF

해당 Project Directory는 다음과 같이 구성되어 있습니다.
- checkpoints: Trainning Model 저장하는 Direcotry
- data
  - Cell Type: GM12878, H1-hESC, HeLa-S3, HepG2, K562
  - TF: CTCF, GABP, JunD, REST, USF2
  - Positive: .fasta, Negative: _back.fasta
- result: Result를 저장하는 Directory
- data_helper.py: Data Preprocessing(One-Hoe-Encoding, batch iter, shuffle)
- data_initialization.py: Data -> One-Hot-Encoding
- utils.py: model hyperparameter initialization, **Gradient Reversal Layer**
- Model
  - Baseline: model_baseline.py
  - DANN_TF: model_DANN.py
- Train
  - Baseline: train_baseline.py
  - DANN_TF: train_DANN.py
- Report: Report.ipynb

<span style="color:red;">전처리 이후의 Data(One-Hoe-Encoding)가 커서 현재 올리지 못하는 상황입니다.</span>

**My Work**  
1. Paper와 다르게 실제 Code에서는 Domain Loss를 Backpropagation하여서 Feature Extractor에 전달해주는 부분에서 𝜆를 사용하지 않고 Constant한 상수값으로 전달하는 부분을 바꾸었습니다.
2. DataPipeline에서 Permutation되는 부분을 삭제하고 다시 구축하였습니다. => data_initialization.py
3. Source Cell Type, Target Cell Type이라고 단순히 2개에 대해서 구별하는 것은 General한 Model이 아니라, Target Cell Type에 Focus를 한 Model이라고 생각하였습니다. 따라서 Domain Classifier에서 모든 CellType에 대하여 구별하도록 Data Pipeline과 Model을 바꾸어서 진행하였습니다. => test_4_cross_cell_type.py <span style="color: #ff0000;">현재 Trainning하고 있어서 결과를 확인하지 못하였습니다.</span>
4. 3의 과정에서 Domain에 대하여 Classification하는 경우 Prediction의 Percentage가 다릅니다. 이에 대하여 Dynamic하게 Data Input으로 넣는 Model을 구상하였으나, <span style="color: #ff0000;">3의 과정이 끝나지 않아서 진행하지 못하였습니다.</span>

**참고**  
1. Paper에서 제공하는 Code에 대한 설명이 잘못된 부분이 있었고 이에 대하여 수정한 부분(𝜆 추가, DataPipeline Permutation, Cross Cell Type과 Label 에 대한 Argument반대로)을 Mail로 Contact해서 알아내려 했으나, 답장이 없는 상태여서 원본 Code가 아닌, 위의 내용대로 진행하였습니다. (실제 Code를 돌려본 결과 위의 수정사항을 포함시켜야지 Paper의 결과와 동일하였습니다.)
2. 1 TF Pairs에 대하여 대략적으로 16시간정도 소요됩니다.

**Reference**  
[1] Domain-Adversarial Training of Neural Networks (https://arxiv.org/pdf/1505.07818.pdf)  
[2] A theory of learning from different domains (http://www.alexkulesza.com/pubs/adapt_mlj10.pdf)  
[3] Cross-Cell-Type Prediction of TF-Binding Site by Integrating Convolutional Neural Network and Adversarial Network (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679139/pdf/ijms-20-03425.pdf)  
[4] Jaejun Yoo's Blog (http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html)
