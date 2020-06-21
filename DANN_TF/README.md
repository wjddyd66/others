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

**Reference**  
[1] Domain-Adversarial Training of Neural Networks (https://arxiv.org/pdf/1505.07818.pdf)  
[2] A theory of learning from different domains (http://www.alexkulesza.com/pubs/adapt_mlj10.pdf)  
[3] Cross-Cell-Type Prediction of TF-Binding Site by Integrating Convolutional Neural Network and Adversarial Network (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679139/pdf/ijms-20-03425.pdf)  
[4] Jaejun Yoo's Blog (http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html)
