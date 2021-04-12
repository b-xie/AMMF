# AMMF: Attention-based Multi-phase Multi-task Fusion Network for Robust 3D Detection in Autonomous Driving


This repository contains the public release of the Python implementation of our AMMF network for 3D object detection.

[**AMMF: Attention-based Multi-phase Multi-task Fusion Network for Robust 3D Detection in Autonomous Driving**]

[Bangquan Xie](https://github.com/b-xie), [Zongming Yang], [Liang Yang], [Ruifa Luo], [Jun Lu], [Ailin Wei], [Xiaoxiong Weng] and [Bing Li]


## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.3.0.

1. Clone this repo
```bash
git clone https://github.com/b-xie/AMMF.git --recurse-submodules
```
If you forget to clone the wavedata submodule:
```bash
git submodule update --init --recursive
```

2. Install Python dependencies
```bash
cd ammf
pip3 install -r requirements.txt
pip3 install tensorflow-gpu==1.3.0
```

3. Add `ammf (top level)` and `wavedata` to your PYTHONPATH
```bash
# For virtualenvwrapper users
add2virtualenv .
add2virtualenv wavedata
```

```bash
# For nonvirtualenv users
export PYTHONPATH=$PYTHONPATH:'/path/to/ammf'
export PYTHONPATH=$PYTHONPATH:'/path/to/ammf/wavedata'
```

4. Compile integral image library in wavedata
```bash
sh scripts/install/build_integral_image_lib.bash
```

5. ammf uses Protobufs to configure model and training parameters. Before the framework can be used, the protos must be compiled (from top level ammf folder):
```bash
sh ammf/protos/run_protoc.sh
```

Alternatively, you can run the `protoc` command directly:
```bash
protoc ammf/protos/*.proto --python_out=.
```

## Training
### Dataset
To train on the [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):
- Download the data and place it in your home folder at `~/Kitti/object`
- Go [here](https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z) and download the `train.txt`, `val.txt` and `trainval.txt` splits into `~/Kitti/object`. Also download the `planes` folder into `~/Kitti/object/training`

The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```
### Customized dataset:

https://drive.google.com/drive/folders/1A-_wfcO_BthlOlGONPXTSVSSX0DcqQPN?usp=sharing

### Run Trainer

Public soon...


### Requirements
matplotlib
numpy>=1.13.0
opencv-python
pandas
pillow
protobuf==3.2.0
scipy
sklearn

## LICENSE
Copyright (c) 2021 [Bangquan Xie](https://github.com/b-xie), [Zongming Yang], [Liang Yang], [Ruifa Luo], [Jun Lu], [Ailin Wei], [Xiaoxiong Weng] and [Bing Li]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgement
Our implementation leverages on the source code from the following repositories:

https://github.com/kujason/avod

