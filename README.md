本项目包括拉土车的一些标注文件，还有几个配置文件，另外还包括一个flask程序用于测试发布，训练好的[模型][1]下载
运行如下：
![demo](https://github.com/sanfooh/yolo_truck/blob/master/demo.png)
## 背景：
> * 1、darknet与yolo,darknet是一个深度学习框架，类似tensorflow,theano,mxnet,caffe之类，包含cnn，rnn等组件，yolo是一个网络模型，类似VGG,resnet之类，yolo原生框架是darknet，yolo也可以使用其它框架来搭建。
> * 2、什么是yolo,yolo主要的作用是实时对象检测：你只看一次（You only look once 的缩写）是一个先进的实时对象检测系统。在Titan X上，它以40-90 FPS的速度处理图像，COCO test-dev上的VOC 2007上的mAP为78.6％，mAP为48.1％。
> * 3、计算机视觉(cv)是深度学习应用的一个重要分支。其应用场景主要有分类，对象检测等。而yolo就要是检测一副图像里有没有存在某些对象，以及这些对象的位置坐标在哪里。
> * 4、darknet是c语言写的，主要依赖opencv和cuda。编译时可以选择这两个或者不选或者只选一个，具体看需要。比如cuda就是决定是否使用GPU，如果需要GPU版的darknet，就要修改编译选项将0改成1，然后编译即可。
> * 5、当我们说yolo时，默认是指基于darknet框架下的yolo，而如果是使用其它框架如tensorflow，应该说是tensorflow版的yolo.

## 训练：
在训练之前，我们先要下载darknet的源代码：

1）如果是linux
```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

如有cuda及opencv，记得修改makefile

2）如果是windows
```
git clone https://github.com/AlexeyAB/darknet.git
```
下载以后根据github上的说明使用vs2015进行编译，编译的时候注意编译选项


### 步骤
> * 1、样本图片与标注：训练yolo时需要大量的图片的样本，当我们提供一张图片时，我们所关注的对象只是图片中一小块区域，所以我们必须把这个区域标注出来，标注信息放在与图片相同目录且文件相同，但后缀名为.txt,比如有一个图片叫truck1.jpg,而你就要提供一个truck1.txt在它旁边，
truck1.txt内容像这样：
```
0 0.491666666666667 0.561611374407583 0.833333333333333 0.872037914691943
```
可能是多行，也可能是一行，一行代表一个对象，多行代表这张图片里有多个对象。它一行的意义如下：
```
<object-class> <x> <y> <width> <height>
```
* object-class：是指对象的索引，从0开始，具体代表哪个对象去obj.names配置文件中按索引查。
* x,y：是一个坐标，需要注意的是它可不是对象左上角的坐标，而对象中心的坐标
* width，height：是指对象的宽高。

> * 2、准备好了一大堆样本图片及其对应的标注文件以后，我们还需要建立两个索引文件，分别叫train.txt及test.txt，名字其实并不重要，它们的意义在于把需要训练的图片路径一张一行的放在train.txt中，而作为验证的图片路径一张一行的放在test.txt中，形如：
 
> * 3、使用darknet来训练yolo模型时，需要三个配置文件，两个索引文件

> > * 1）模型配置文件，比如名叫my-yolo-net.cfg,需要将yolo的模型结构写到一个配置文件如:
```
[net]
#Testing
batch=64
subdivisions=8
#Training
#batch=64
#subdivisions=8
height=416
width=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 80200
policy=steps
steps=40000,60000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
//省略


其中像convolutional节就代表一个卷积层，指定有多少个卷积核心filters=30，基本上就是一个网络结构定义以及一些其它配置。
```
> > * 2)一个对象名称文件，比如叫obj.names，这个文件比较简单，一行一个对象名称：
dog
fox
cat
.....

> > * 3)最后还有一个obj.data文件：
```
classes= 1
train = /darknet/work/train.txt
valid = /darknet/work/test.txt
names = /darknet/work/cfg/obj.names
backup = /output/
```
* classes是指对象分类，这里我只检测一类对象，所以是1
* names就是指对象名称文件
* backup是指darknet在训练到比如100轮，200轮....时会自动把模型保存到该目录。它产生的文件名形如：YOLO-obj_9000.* weight，一看到这个文件名，即可知道已经训练了几轮了。
* train 和valid 是指索引文件了。

>  * 4、在训练之前，还要再下载一个预训练文件：darknet19_448.conv.23（https://pjreddie.com/media/files/darknet19_448.conv.23），然后执行命令：
```
darknet.exe detector train cfg/obj.data cfg/yolo-obj.cfg darknet19_448.conv.23
```
> > * darknet.exe detector train：这一段是命令本身，说要开始训练了。
> > * cfg/obj.data：就是上面说的obj.data,里有指定train.txt,test.txt，obj.names,backup的路径
> > * cfg/yolo-obj.cfg：就是模型的网格结构文件。
> > * darknet19_448.conv.23：是预训练文件，预训练文件是让我们基于别人的基础上更快的训练出自己的模型，这样会比较省时间。

>* 5、在darknet源代码里，可以找到很多的配置文件示例，需要花点时间去观赏一下。

>  * 6、训练了一段时间以后，忽然关机了，怎么办？没事，再次开机以后去backup目录里找一找，一般会找到如：yolo-obj_1000.weights，yolo-obj_2000.weights这类文件，然后使用最后一个
darknet.exe detector train cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_2000.weights
这样，它又会继续往下训练，而不会重头开始。

>  * 7、训练过程长这样:
那么什么情况下觉得可以停止了呢？主要是看IOU，如果IOU已经接近于1，那么说明不错了。
这边已经训练了一个泥头车辆的[模型][1]
## 使用：
当我们训练完，一般在普通GPU上要花费几个小时，产生了很多的模型文件以后，我们就可以验证模型了，首先是用命令行来验证：
```
darknet.exe detector test cfg/obj.data cfg/yolo-obj.cfg yolo-obj1000.weights data/testimage.jpg
```
>  * 1、当验证完需要布署到服务上时，我们可以：
>  * 2）linux上使用darknet.so库，使用接口编写自己的程序。
>  * 3）使用项目中的darknet.py文件，利用flask发布出来(我写的例子如下：https://github.com/sanfooh/yolo_truck)。其中flask部分来自https://github.com/makefile/objdet_web
>  * 4）在windows上直接修改项目文件，改成darknet.dll使用。


### 参考：
>  * https://pjreddie.com/darknet/yolo/
>  * https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
>  * https://github.com/makefile/objdet_web

[1]:https://www.floydhub.com/sansooh1/datasets/quick-start/1