# MTCNN_tf2.0

### Dataset


WiderFace : http://shuoyang1213.me/WIDERFACE/

FaceLandmarks : http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

First : download解壓縮放在./data/

And Then : 用preprocessing裡的code去做pnet、rnet、onet的training data

Note:生的data要控在 pos:part:neg = 1:1:3 , 不然model會always output no boxes, 多的data從txt檔delete就可以了


### PNet Output Result
![PNet_output_boxes](./test_imgs/result_pnet.jpg)
