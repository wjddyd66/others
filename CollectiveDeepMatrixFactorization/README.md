### Project
GitHub: https://github.com/wjddyd66/others/tree/master/CollectiveDeepMatrixFactorization

해당 Project Directory는 다음과 같이 구성되어 있습니다.
- SNP_Data_Preprocessing.ipynb: SNPs Data Preprocessing
- SNP_DATA: SNP_Data_Preprocessing 결과 Data
- MRI_Data_Preprocessing.ipynb: MRI Data Preprocessing
- MRI_DATA: MRI_Data_Preprocessing 결과 Data
- Merge_DATA: Mri preprocessing output(MRI_DATA)와 SNP Preprocessing output(SNP_DATA)를 Merge하여 실제 Model에 넣기위한 Merge 및 Preprocessing
- data
  - NL_AD_Data: NL_AD의 Model에 들어가는 Data및 결과
  - NL_MCI_Data: NL_MCI의 Model에 들어가는 Data및 결과
  - NL_MCI_AD_Data: NL_MCI_AD의 Model에 들어가는 Data 및 결과
- Algorithm.ipynb: 해당 Model을 만들때 참고한 Algorithm
- Project_Model.ipynb: Model Trainning

<span style="color:red;">전처리 이전의 Data의 경우 Data가 매우 커서 현재 올리지 못하는 상황입니다.</span>
- MRI: 150GB
- SNPs: 4GB

**Reference**  
[1] Multi-Modality Disease Modeling via Collective Deep Matrix Factorization (https://dl.acm.org/doi/pdf/10.1145/3097983.3098164)  
[2] SVD based initialization: A head start for nonnegative matrix factorization (https://www.sciencedirect.com/science/article/pii/S0031320307004359)  
[3] The Effect of Age Correction on Multivariate Classification in Alzheimer’s Disease, with a Focus on the Characteristics of Incorrectly and Correctly Classified Subjects (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4754326/)  
[4] A large scale multivariate parallel ICA method reveals novel imaging–genetic relationships for Alzheimer's disease in the ADNI cohort's Method (https://pubmed.ncbi.nlm.nih.gov/22245343/)

**Tools**  
사용한 툴은 다음과 같습니다.  
[1] FreeSurfer (https://surfer.nmr.mgh.harvard.edu/)  
[2] PLINK (https://www.cog-genomics.org/plink/)

**Library**  
Project에서 사용한 Library는 requirements.txt에 있습니다.

**Inference**  
일반적인 Model과 다르게 MatrixFactorization은 Data로부터 Matrix를 만들고 이것을 Weight로 생각하여 Weight끼리 연산을하여 Update하는 형식입니다. 따라서, 사용한 Weight를 모두 Result Directory에 저장하였습니다. 또한, 실제 결과는 load_parameter_test.py로서 실행할 수 있습니다.
