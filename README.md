# colav-simulator: PSB-MPC Setup Guide
This README details how the PSB-MPC can be utilized as a COLAV system within the simulator. The PSB-MPC is implemented in the [Collision Avoidance algorithm repository](https://github.com/NTNU-Autoship-Internal/thecolavrepo), and it can, from the simulator, be set to utilize the Intention Model (IM), which is implemented in the [Ship Intention Inference repository](https://github.com/NTNU-Autoship-Internal/ship_intention_inference).


[![platform](https://img.shields.io/badge/platform-linux-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.11-blue)]()

## 1. Setup and Test the colav-simulator on the main branch
First, before you start this setup procedure, ensure that your setup of the colav-simulator works by completing the setup procedures and tests which are outlined on the main branch.


## 2. CMake and Python Environment
If you do not already have CMake or a virtual Python environment set up on your system, run the following commands. Ensure that your CMake version is greater than or equal to 3.22, and that you are using Python 3.11. You should either use the Python environment you already set up to use in the colav-simulator or set up a new one that mirrors the one you are already using on the main branch.


```bash
sudo apt install cmake
cmake --version

conda create -n <venvname> python=3.11
conda activate <venvname>
```

## 3. Setting up the PSB-MPC branch
The PSB-MPC relies on submodules. Use the following commands to start the setup process of these.

```bash
git fetch origin feature/psbmpc
git checkout feature/psbmpc
git submodule update --init --recursive
git submodule update --remote --recursive
```

## 4. Setting up PyBind11
The PSB-MPC interface was built with [PyBind11](https://github.com/pybind/pybind11). It allows Python to use compiled C++ code. From the root of the colav-simulator, run the following commands to set up PyBind11. The two last commands can be ran to verify the PyBind11 installation.

```bash
cd colav_simulator/core/colav/cpp_to_py_interfaces/external/pybind11
cmake -S . -B build
cmake --build build
cmake --install build

conda install -c conda-forge pybind11

cmake -S . -B build --debug-find
pip show pybind11
```

## 5. PSB-MPC and IM Dependencies
The following depndencies are required to be able to compile the PSB-MPC and IM C++ code. The last three commands can be ran to verify the installations.

```bash
sudo apt -y install libboost-all-dev
sudo apt -y install libgeographic-dev
sudo apt -y install libeigen3-dev

dpkg -l libboost-all-dev
dpkg -l libgeographic-dev
dpkg -l libeigen3-dev
```

The [Ship Intention Inference repository](https://github.com/NTNU-Autoship-Internal/ship_intention_inference) requires further setup. The C++ SMILE library and a SMILE Academic license, that is `smile_license.h`, must be downloaded from the [Bayesfusion website](https://download.bayesfusion.com/files.html?category=Academia) and extracted into the `/smile` directory. It should be inside the `/external` directory in the repository.

 ```bash
cd ../ship_intention_inference/external
mkdir smile
cd smile
```

## 6. Compiling the Interface/Wrapper: cpp_to_py_interfaces
The last step is to compile the interface (which itself compiles the PSB-MPC and the IM code). Whenever changes are made to the C++ code, that is the [pybind_im_and_psbmpc interface](https://github.com/NTNU-Autoship-Internal/pybind_im_and_psbmpc/tree/main), the [Collision Avoidance algorithm repository](https://github.com/NTNU-Autoship-Internal/thecolavrepo), or the [Ship Intention Inference repository](https://github.com/NTNU-Autoship-Internal/ship_intention_inference), the code must be recompiled. To compile the code navigate to the `/cpp_to_py_interfaces` directory in the colav-simulator. If CMake finds the wrong Python version you can try to use `cmake -S . -B build -DPython_EXECUTABLE=$(which python)` instead of `cmake -S . -B build`.

 ```bash
cd ../../../../ # cd to cpp_to_py_interfaces
cmake -S . -B build 
cmake --build build
```

## Testing the PSB-MPC
A test file, `test_psbmpc.py`, can be found inside the `/tests` directory of the colav-simulator. Run this to verify that your installation has been successful.
