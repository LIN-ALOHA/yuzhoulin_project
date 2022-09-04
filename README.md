# yuzhoulin_project
this repo is about demo and evaluation on interpreting image-based malware dataset on machine-learning by varities of explaining methods.
## background
the project is aimed at proposing an interpreble image-based malware detection architecture called IEMD, namely image-based ensemble malware detection.
Currently, deep-learning-based methods towards malware detection lack interpretability, which is far from practising them in the real industry area.
Our work aims at resolving and alleviating this bad trend and simultaneously creating some useful metrics for evaluating malware detection interpretability quantitatively.
## usage
By operating a demo on our work, you need to firstly employ comprehensive.py file for the further use.
```
  python comprehensive.py -[args]
```
### methods
We have major methods for explaining deep-learning malware detection, namely, deep-lift, guided-g
rad CAM, lemna, and smooth-grad.
### datasets
We use Malimg, IoT malware, for more information, please contact:
https://pan.baidu.com/s/1qXfiPg_QE_t46dlsG5tbWg
passwd: 37eq 
Inside our entire work, in baidu cloud dst, we have everything required for you to conduct an entire demo for deeply understand our work.
### IEMD
####contribution
*ensemble-learning architecture
*deep Taylor decomposition embedded on ensemble structure with/without reweighting technique
*iDrop normalization technique here adjusted the parameter with 0.6, according to the tuning process depicted in ariticle
Lin, Y., & Chang, X. (2021). Towards Interpretable Ensemble Learning for Image-based Malware Detection. arXiv preprint arXiv:2101.04889.
*detailed tuning curves here
![image]()
####our framework
![image](https://github.com/LIN-ALOHA/yuzhoulin_project/blob/lester/IEMD_framework.png)
####our metrics and relative evaluation experiment results
methods|accuracy|precision rate|recall rate|f1-score
------|-------|-------|----|----
IMCFN(DL)|0.98|0.98|0.98|0.98
dl|0.97|n/a|0.88|n/a
knn on GIST features|0.97|n/a|n/a|n/a
cnn+svm on entropy graphs|0.99|n/a|n/a|0.99
IEMD(our work)|0.99|0.99|0.99|0.99
