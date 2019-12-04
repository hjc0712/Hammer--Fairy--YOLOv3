# Hammer--Fairy--YOLOv3
The real time object detection project by hammer &amp; fairy team

## Preperations before running
Before running the project, need to get the following steps done first.  
1) We are using the PASCAL VOC datasets provided on mldsp server. But we need to move data into certain folders, and change labels into YOLO syntax.
2) Move the images of your dataset to data/custom/images/ , by running:  
```python
 cp /datasets/ee285f-public/PascalVOC2012/JPEGImages/* data/custom/images/
``` 
3) Trasformed VOC annotations into YOLOv3 supported labels, which is in syntax "label_idx x_center y_center width height". To convert the labels and generate manipast file at the same time, run "convert.py":
```python
python3 convert.py --datasets VOC --img_path /datasets/ee285f-public/PascalVOC2012/JPEGImages/ --label /datasets/ee285f-public/PascalVOC2012/Annotations/ --convert_output_path data/custom/labels/ --img_type ".jpg" --manipast_path ./ --cls_list_file config/voc.txt
```
4) Finally, check the following configrations and make sure everything's right.  
a. custom.data
```python
classes= 20
train=config/custom_train.txt
valid=config/custom_test.txt
names=config/voc.txt
```  
b. data/custom/images & data/custom/labels are in the right directory and contain the required information. Labels should be in this syntax:
```python
12 0.502 0.625 0.076 0.168
14 0.514 0.555 0.1 0.192
11 0.332 0.829 0.06 0.149
11 0.413 0.795 0.03 0.064
```
c. custom_train.txt & custom_test.txt & voc.txt are in the right directory and contain the correct information.

5) Now we are ready to staring trainning & testing. run:
```python
train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights checkpoints_voc/yolov3_ckpt_52.pth 2> /dev/null
```  
replace the checkpoint with the one we want to start. 
