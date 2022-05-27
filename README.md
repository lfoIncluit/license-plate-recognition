# license-plate-recognition
The license-plate-recognition call the OpenVINO pre-trained models with python and the DNN in OpenCV. It comes from the [Security Barrier Camera Demo](https://docs.openvinotoolkit.org/2018_R5/_samples_security_barrier_camera_demo_README.html).This demo showcases Vehicle and License Plate Detection network followed by the Vehicle Attributes Recognition and License Plate Recognition networks applied on top of the detection results
* vehicle-license-plate-detection-barrier-0106, which is a primary detection network to find the vehicles and license plates
* license-plate-recognition-barrier-0007, which is executed on top of the results from the first network and reports a string per recognized license plate

# How to run the license-plate-recognition
 python3 testOpenVINO.py
