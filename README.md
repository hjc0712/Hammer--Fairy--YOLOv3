# Hammer--Fairy--YOLOv3
The real time object detection project by hammer &amp; fairy team

## Requirements
Python 3 (>= 3.7.0)

## Preperations before running
1. We are using the PASCAL VOC datasets provided on mldsp server. But we need to move data into certain folders, and change labels into YOLO syntax.
2. Move the images of your dataset to data/custom/images/ , by running:  
```python
 cp /datasets/ee285f-public/PascalVOC2012/JPEGImages/* data/custom/images/
``` 
3. Trasformed VOC annotations into YOLOv3 supported labels, which is in syntax "label_idx x_center y_center width height".
```python
python3 convert.py --datasets VOC --img_path data/custom/images/ --label /datasets/ee285f-public/PascalVOC2012/Annotations/ --convert_output_path data/custom/labels --img_type ".jpg" --cls_list_file data/custom/classes.names
```
4. Finally, check the following configrations and make sure everything's right.  
a. custom.data
```python
classes= 20
train=data/custom/train.txt
valid=data/custom/valid.txt
names=data/custom/classes.names
```  
b. data/custom/images & data/custom/labels are in the right directory and contain the required information. Labels should be in this syntax:
```python
12 0.502 0.625 0.076 0.168
14 0.514 0.555 0.1 0.192
11 0.332 0.829 0.06 0.149
11 0.413 0.795 0.03 0.064
```
c. train.txt & valid.txt & classes.names are in the right directory and contain the correct information.

## Training
Now we are ready to staring trainning. run:
```python
python3 -W ignore train.py --pretrained_weights weights/darknet53.conv.74 --batch_size 2
```  
replace the checkpoint with the one we want to start with.

## Testing
To run test, for example, run a test on the model we trained:
```python
python3 test.py --weights_path checkpoints/yolov3_ckpt_309.pth
```

## Detection Demo
To see how well our model works, open demo.ipynb as a jupyter notebook and run the first cell. You should see detections and related classification information for random images.