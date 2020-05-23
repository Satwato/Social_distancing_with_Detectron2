# Social_distancing_with_Detectron2

## Modules needed:
```
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
pip install scipy
pip install numpy

```
You will also need opencv
Make sure your python version is 1.4+


## How does it work

I used detectron2 to get the predictions and used my own visualizing module (`image_viz.py`) to bound-box only humans. The later part of the code which deals with social distance was completed with the help of [this article](https://www.analyticsvidhya.com/blog/2020/05/social-distancing-detection-tool-deep-learning/) which I found way later after starting this project and realised that they also have used Detectron2 for their detection (If I had found this article slightly earlier, it would have saved a lot of time). The main inspiration for this article came from this [medium post](https://medium.com/@drojasug/measuring-social-distancing-using-tensorflow-object-detection-api-7c54badb5092)

## How to run the code

First, it is highly suggestible to download the [model](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl) and keep it in the main folder.

 
```
python runner.py --config-file <path_detectron2_repo>/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--input cctv-footage.jpg  --opts MODEL.DEVICE cpu  MODEL.WEIGHTS model_final_f10217.pkl 
```

* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.
