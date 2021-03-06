# MTCNN_tf2.0

### Paper

* implement paper : 
     
    * multi-task cnn : https://arxiv.org/abs/1604.02878

* related paper : 

    * prelu : https://arxiv.org/abs/1502.01852
    
    * cnn cascade structure : https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf

    * adverserial attack : https://arxiv.org/pdf/1910.06261.pdf


### Dataset


* WiderFace : http://shuoyang1213.me/WIDERFACE/

* FaceLandmarks : http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

* First : download解壓縮放在./data/

* And Then : 用preprocessing裡的code去做pnet、rnet、onet的training data

* Note : 

    * Pnet生的data要大概控在 pos:part:neg = 1:1:3 , 不然model會always output no boxes (part、neg太多,大部分的時間都在compute neg bce跟box regression) , 多的data從txt檔delete就可以了
    
    * RNet跟ONet遵守imbalance data的控制就好了(pos:neg = 1:1) 
    
    * 我自己的意見 : Train RNet不用一定要PNet的output
    
    * 我只train了 100000:100000:300000個data, 因為我的computer受不了

    * small tips: each epoch do data augmentation when pick a batch of imgs


### PNet Output Result
![pnet](./test_imgs/gsw3_result_pnet.jpg)


### RNet Output Result
![rnet](./test_imgs/gsw3_result_rnet.jpg)



### ONet Output Result
![onet](./test_imgs/gsw3_result_output.jpg)

### Video Test

yt : https://www.youtube.com/watch?v=D5SlXduGc34



