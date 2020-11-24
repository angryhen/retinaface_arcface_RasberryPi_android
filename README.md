# **retinaface_arcface_RasberryPi_android**

---

## Model Download

https://pan.baidu.com/s/1ZO8Y_wOsXHBVKmynghne4Q 
password:zqsz 

> unzip model.zip into directory

## Run

make sure you have changed  `project_path` to your own

```shell
mkdir build
cd build
cmake ..
make -j4
./LiveFaceReco
```

## Android

### step1

https://github.com/Tencent/ncnn/releases

download ncnn-android-vulkan-lib.zip or build ncnn for android yourself

### step2

extract ncnn-android-vulkan-lib.zip into app/src/main/jni or change the ncnn path to yours in app/src/main/jni/CMakeLists.txt

### step3

open this project with Android Studio, build it and enjoy!

## Citation

> ```
> @inproceedings{deng2018arcface,
> title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
> author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
> booktitle={CVPR},
> year={2019}
> }
> 
> @inproceedings{deng2019retinaface,
> title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
> author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
> booktitle={arxiv},
> year={2019}
> }
> 
> @inproceedings{ncnn,
> title={ncnn https://github.com/ElegantGod/ncnn},
> author={ElegantGod},
> }
> 
> @inproceedings{Face-Recognition-Cpp,
> title={Face-Recognition-Cpp https://github.com/markson14/Face-Recognition-Cpp},
> author={markson14},
> }
> 
> @inproceedings{insightface_ncnn,
> title={insightface_ncnn https://github.com/KangKangLoveCat/insightface_ncnn},
> author={KangKangLoveCat},
> }
> 
> @inproceedings{Silent-Face-Anti-Spoofing,
> title={Silent-Face-Anti-Spoofing https://github.com/minivision-ai/Silent-Face-Anti-Spoofing},
> author={minivision-ai},
> }
> ```
>

## References

Appreciate the great work from the following repositories:

https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi

https://github.com/nihui/ncnn-android-yolov5







