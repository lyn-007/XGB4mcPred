# XGB4mcPred

Identification of DNA N4-methylcytosine Sites in multiple species based on an eXtreme Gradient Boosting algorithm and DNA sequence information


## Abstract
<p align="justify">DNA N4-methylcytosine(4mC) plays an important role in numerous biological func-tions and is a mechanism of particular epigenetic importance. Therefore, accurate identification of the 4mC sites in DNA sequences is necessary to understand the functional mechanism. Although some effective calculation tools have been proposed to identifying DNA 4mC sites, it is still chal-lenging to improve identification accuracy and generalization ability. Therefore, it is a great need to build a calculation tool to accurately identifying the position of DNA 4mC sites. Hence, this study proposed a novel predictor XGB4mcPred, a predictor for 4mC sites identification trained using an extreme Gradient Boosting algorithm (XGBoost) and DNA sequence information. Firstly, we use the One-Hot encoding on adjacent and spaced nucleotide, dinucleotide and trinucleotide of the original 4mC sites sequences as feature vectors. Then, the importance values of the feature vectors of pre-train by XGBoost algorithm are used as a threshold to filter redundant features, resulting in a significant improvement in the identification accuracy of the constructed XGB4mcPred predictor to identify 4mC sites. The analysis showed that there is a clear preference for nucleotide sequences between 4mC sites and non-4mC sites sequences in six datasets from multiple species, and the optimized features can better distinguish 4mC sites from non-4mC sites. Additionally, XGB4mcPred also showed the results of cross-validation and independent test from six different species have been improved to varying degrees compared to other state-of-the-art predictors. The user-friendly webserver that the XGB4mcPred predictor has been developed and can be accessed for free at http://www.xwanglab.com/XGB4mcPred/.</p>

## Keywords

N4-methylcytosine; eXtreme Gradient Boosting; sites identification; feature selection.

## Datasets

### Benchmark Datasets
- A. thaliana: [A_tha.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/A_tha.txt)
- C. elegans: [C_ele.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/C_ele.txt)
- D. melanogaster: [D_mel.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/D_mel.txt)
- E. coli: [E_col.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/E_col.txt)
- G. pickeringii: [G_pic.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/G_pic.txt)
- G. subterruneus: [G_sub.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Benchmark%20Datasets/G_sub.txt)


### Independent Datasets
- A. thaliana: [A_tha_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/A_tha_indep.txt)
- C. elegans: [C_ele_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/C_ele_indep.txt)
- D. melanogaster: [D_mel_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/D_mel_indep.txt)
- E. coli: [E_col_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/E_col_indep.txt)
- G. pickeringii: [G_pic_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/G_pic_indep.txt)
- G. subterruneus: [G_sub_indep.txt](https://github.com/lyn-007/XGB4mcPred/blob/main/Independent%20Datasets/G_sub_indep.txt)

## PKL

- [features_pkl](https://github.com/lyn-007/XGB4mcPred/blob/main/features_pkl): The optimal features for each datasets are stored in this folder.
- [pkl](https://github.com/lyn-007/XGB4mcPred/blob/main/pkl): The optimal model for each datasets are stored in this folder.

## Code




