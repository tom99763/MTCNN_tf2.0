# MTCNN_tf2.0

### Paper

* implement paper : 
     
    * multi-task cnn : https://arxiv.org/abs/1604.02878

* related paper : 

    * prelu : https://arxiv.org/abs/1502.01852
    
    * cnn cascade structure : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf


### Dataset


* WiderFace : http://shuoyang1213.me/WIDERFACE/

* FaceLandmarks : http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

* First : download解壓縮放在./data/

* And Then : 用preprocessing裡的code去做pnet、rnet、onet的training data

* Note : 

    * Pnet生的data要大概控在 pos:part:neg = 1:1:3 , 不然model會always output no boxes (part、neg太多,大部分的時間都在compute neg bce跟box regression) , 多的data從txt檔delete就可以了
    
    * RNet跟ONet遵守imbalance data的控制就好了(pos:neg = 1:1)
    
    * Note : 我只train了 100000:100000:300000個data, 因為我的computer受不了


