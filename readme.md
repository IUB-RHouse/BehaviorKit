# Setup

## Bundled models

Face detection (YOLOv8-Face) and face/pose landmarks (MediaPipe) ship with
this repo — their weights are already committed under `data/models/` and
need no separate download:

| modality                          | model                          | weights                                                          | notes |
|------------------------------------|---------------------------------|-------------------------------------------------------------------|-------|
| `face_detection`                   | YOLOv8m-Face                    | `data/models/yolov8-face/yolov8m-face.pt`                         | MIT-licensed ([lindevs/yolov8-face](https://github.com/lindevs/yolov8-face)). |
| `face_landmarks`                   | MediaPipe Face Landmarker (FaceMesh V2) | `data/models/mediapipe/face_landmarker.task`               | Apache 2.0, from Google. |
| `pose_landmarks`                   | MediaPipe Pose Landmarker (BlazePose), lite/full/heavy | `data/models/mediapipe/pose_landmarker_{lite,full,heavy}.task` | Apache 2.0, from Google. Variant picked via `modalities.pose_landmarks.variant`, or derived from `quality` if left `null`. |

Gaze estimation is the one modality with a pluggable, user-selected backend —
see below. Emotion recognition also ships a bundled weight, but is off by
default because of its license — see below.

## Emotion recognition (optional)

`modalities.emotion` runs [EmoNet](https://github.com/face-analysis/emonet)
on the same face crop gaze uses, producing an 8-class expression
(neutral/happy/sad/surprise/fear/disgust/anger/contempt) plus continuous
valence/arousal. The weights ship in the repo at
`data/models/emonet/emonet_8.pth` — no download needed.

**License:** EmoNet is CC BY-NC-ND 4.0 (see `data/models/emonet/LICENSE` and
`NOTICE`) — **non-commercial use only, no derivative redistribution.** The
model code is vendored unmodified in `module/emonet_backend/emonet.py`.
Because of this, `emotion.enabled` defaults to `false` in `config.yaml`; you
must opt in, and you're responsible for staying within the license terms
(e.g. don't enable it in a commercial deployment). Citation required in
publications:
A. Toisoul, J. Kossaifi, A. Bulat, G. Tzimiropoulos, M. Pantic,
"Estimation of continuous valence and arousal levels from faces in
naturalistic conditions", Nature Machine Intelligence, 2021.

Like gaze, emotion needs `face_detection` enabled to get a face crop.

## Gaze estimation backend

The gaze model is pluggable — pick one via `gaze.method` in `config.yaml`:

| method       | model                                              | weights                                                                                                                   | notes |
|--------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-------|
| `gaze360`    | ResNet18 + BiLSTM (original Gaze360 GazeLSTM)        | `data/models/gaze360/gaze360_model.pth` — **gated**, see below                                                          | Official MIT model. As of writing, MIT's public download link for the weights is offline; you must obtain them separately. |
| `l2cs`       | L2CS-Net, ResNet50, trained on Gaze360                | `data/models/l2cs/L2CSNet_gaze360.pkl` — download from the [official Google Drive folder](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd) and place at that path | MIT-licensed code ([Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net)); not redistributed here. |
| `mobilegaze` | ResNet18/34/50 or MobileNetV2, trained on Gaze360 (default: MobileNetV2) | `data/models/mobilegaze/<arch>.pt` — downloadable directly, e.g. `curl -L -o data/models/mobilegaze/mobilenetv2.pt https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2.pt` | MIT-licensed, built on L2CS-Net ([yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)). Lightest option, good fit for small GPUs. **Default.** |

Only the weights for the selected `gaze.method` need to be present.

### Gaze360 (original model)

Download and extract: http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar -> (location within this repo) data/models/gaze360/gaze360_model.pth

Due to the Gaze360 Research License, we do not redistribute the weights.
Please obtain them from the official source after accepting the license (https://gaze360.csail.mit.edu/):

Paper for Gaze360 (they require its citation in publications):
P. Kellnhofer, A. Recasens, S. Stent, W. Matusik, A. Torralba.
"Gaze360: Physically Unconstrained Gaze Estimation in the Wild", ICCV 2019.

* This guide assumes you are using Ubuntu (we are using Ubuntu 22.02) and you have installed Python 3.10. We additonally assume you have CUDA installed (we had CUDA 12.8, to match the `torch==2.9.1+cu128` pin in server_requirements.txt) and therefore also have an NVIDIA GPU. While it is possible to run the models on the CPU, we have currently disabled that feature (you may reenable that feature if you so choose to do so).

* Please note, that the client and server are supposed to be seperate computers.

# Installation

Please navigate inside of the folder containing the BehaviorKit code.

## Installing the BehaviorKit client (Ubuntu)
    python3.10 -m venv client_venv
    source client_venv/bin/activate
    pip install -r client_requirements.txt

## Installing the BehaviorKit server (Ubuntu)
    python3.10 -m venv server_venv
    source server_venv/bin/activate
    pip install -r server_requirements.txt

* Please note that the default port is 8010 and the address is localhost. These can be modified by going to client_tools/config.py and server.py respectively. The server must be able to be connected to in order for the client to be able to run.

# Running the server for demo (Ubuntu)
    source server_venv/bin/activate
    python server.py

# Running the client for demo (Ubuntu)
    source client_venv/bin/activate
    python client.py

# Running tests (Ubuntu)
    source server_venv/bin/activate
    pytest