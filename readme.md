# Setup

## First Steps: Install Gaze360 (https://gaze360.csail.mit.edu/)

Download and extract: http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar -> (location within this repo) data/models/gaze360/gaze360_model.pth

Due to the Gaze360 Research License, we do not redistribute the weights.
Please obtain them from the official source after accepting the license:

Paper for Gaze360 (they require its citation in publications):
P. Kellnhofer, A. Recasens, S. Stent, W. Matusik, A. Torralba.
"Gaze360: Physically Unconstrained Gaze Estimation in the Wild", ICCV 2019.

* This guide assumes you are using Ubuntu (we are using Ubuntu 22.02) and you have installed Python 3.10. We additonally assume you have CUDA installed (we had CUDA 12.1) and therefore also have an NVIDIA GPU. While it is possible to run the models on the CPU, we have currently disabled that feature (you may reenable that feature if you so choose to do so).

* Please note, that the client and server are supposed to be seperate computers.

# Installation

Please navigate inside of the folder containing the BehaviorKit code.

## Installing the BehaviorKit client (Ubuntu)
    python3.10 -m venv server_venv
    source server_venv/bin/activate
    pip install -r client_requirements.txt

## Installing the BehaviorKit server (Ubuntu)
    python3.10 -m venv client_venv
    source client_venv/bin/activate
    pip install -r server_requirements.txt

* Please note that the default port is 8010 and the address is localhost. These can be modified by going to client_tools/config.py and server.py respectively. The server must be able to be connected to in order for the client to be able to run.

# Running the server for demo (Ubuntu)
    source server_venv/bin/activate
    python server.py

# Running the client for demo (Ubuntu)
    source client_venv/bin/activate
    python client.py