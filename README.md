# yolo_truck

#1、floyd客户端操作


floyd login --username aa --password bb

H:


cd H:\deeplearning\floydhub\yolo1


floyd init quick-start


floyd run --data sanfooh15/datasets/darknet19_448conv23:/data --gpu --mode jupyter


#2、打开jupyter notebook terminal


cd ..


git clone https://github.com/sanfooh/darknet_for_floydhub.git


mv darknet_for_floydhub darknet


cd darknet


git clone https://github.com/sanfooh/yolo_truck.git


mv yolo_truck work


./darknet detector train work/cfg/obj.data work/cfg/YOLO-obj.cfg /data/darknet19_448.conv.23
