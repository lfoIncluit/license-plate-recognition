# license-plate-recognition
The license-plate-recognition call the OpenVINO pre-trained models with python and the DNN in OpenCV. It comes from the [Security Barrier Camera Demo](https://docs.openvinotoolkit.org/2018_R5/_samples_security_barrier_camera_demo_README.html).This demo showcases Vehicle and License Plate Detection network followed by the Vehicle Attributes Recognition and License Plate Recognition networks applied on top of the detection results
* vehicle-license-plate-detection-barrier-0106, which is a primary detection network to find the vehicles and license plates
* vehicle-attributes-recognition-barrier-0039, which is executed on top of the results from the first network and reports general vehicle attributes, for example, vehicle type (car/van/bus/track) and color
* license-plate-recognition-barrier-0007, which is executed on top of the results from the first network and reports a string per recognized license plate

# How to run the license-plate-recognition
1. OpenVINO installation and environment configuration. More information Please refer to [OpenVINO+Win10安装及环境配置攻略](https://livezingy.com/setup-openvino-in-win10/).
2. Run the setupvars.bat in C:\Program Files (x86)\Intel\openvino_2021.1.110\bin
3. Run the testOpenVINO.py. The license-plate-recognition will process the images in testImgs one-by-one.
```
C:\Users\username>cd C:\Program Files (x86)\Intel\openvino_2021.1.110\bin
 
C:\Program Files (x86)\Intel\openvino_2021.1.110\bin>setupvars.bat
Python 3.6.5
[setupvars.bat] OpenVINO environment initialized
 
C:\Program Files (x86)\Intel\openvino_2021.1.110\bin>D:
 
D:\>cd D:\Python\OpenVINOMODEL
 
D:\Python\OpenVINOMODEL>python testOpenVINO.py
[E:] [BSL] found 0 ioexpander device
```
![image](https://livezingy.com/uploads/2020/12/plate3.png)

![image](https://livezingy.com/uploads/2020/12/plate4.png)

# vehicle-attributes-recognition-barrier-0039 was removed.

# More Information
[OpenVINO+Win10安装及环境配置攻略](https://livezingy.com/setup-openvino-in-win10/)

[OpenCV调用OpenVINO模型vehicle-license-plate-detection-barrier检测车牌](https://livezingy.com/vehicle-license-plate-detection-barrier-in-opencv/)

