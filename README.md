# XGB4mcPred: Identification of DNA N4-methylcytosine Sites in multiple species based on an eXtreme Gradient Boosting algorithm and DNA sequence information


<p align="justify"><strong>Abstract:</strong> DNA N4-methylcytosine(4mC) plays an important role in numerous biological func-tions and is a mechanism of particular epigenetic importance. Therefore, accurate identification of the 4mC sites in DNA sequences is necessary to understand the functional mechanism. Although some effective calculation tools have been proposed to identifying DNA 4mC sites, it is still chal-lenging to improve identification accuracy and generalization ability. Therefore, it is a great need to build a calculation tool to accurately identifying the position of DNA 4mC sites. Hence, this study proposed a novel predictor XGB4mcPred, a predictor for 4mC sites identification trained using an extreme Gradient Boosting algorithm (XGBoost) and DNA sequence information. Firstly, we use the One-Hot encoding on adjacent and spaced nucleotide, dinucleotide and trinucleotide of the original 4mC sites sequences as feature vectors. Then, the importance values of the feature vectors of pre-train by XGBoost algorithm are used as a threshold to filter redundant features, resulting in a significant improvement in the identification accuracy of the constructed XGB4mcPred predictor to identify 4mC sites. The analysis showed that there is a clear preference for nucleotide sequences between 4mC sites and non-4mC sites sequences in six datasets from multiple species, and the optimized features can better distinguish 4mC sites from non-4mC sites. Additionally, XGB4mcPred also showed the results of cross-validation and independent test from six different species have been improved to varying degrees compared to other state-of-the-art predictors. The user-friendly webserver that the XGB4mcPred predictor has been developed and can be accessed for free at http://www.xwanglab.com/XGB4mcPred/.</p>


<strong>Keywords</strong>: N4-methylcytosine; eXtreme Gradient Boosting; sites identification; feature selection.
