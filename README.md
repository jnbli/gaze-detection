# Gaze-Detection

## About
The code is based on the Gaze controlled keyboard tutorials from PySource. I added face turn direction.

## Installation
```bash
pip install numpy
pip install opencv-python
pip install cmake
pip install dlib
```

## How To Run
```bash
python main.py
```

## What To Expect
![Facing Forward](./img/eye_center_face_center.png)
![Left Glean](./img/eye_left_face_center.png)
![Blinking](./img/blinking.png)
![Face Right](./img/eye_right_face_right.png)

## References
For facial landmark points visit [OpenFace API Doc](https://openface-api.readthedocs.io/en/latest/openface.html).
For PySource Tutorial visit [Gaze controlled keyboard with Opencv and Python YouTube Playlist](https://www.youtube.com/playlist?list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8).
