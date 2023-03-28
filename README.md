# Ray - examples

## Introduction

This repository contains some python scripts demonstrating ray framework for distributed computing & machine learning.


## Prerequisites

- Python3 (3.6 or higher)

- Install requirements
    ```
    $ pip install -r requirements.txt
    ```

## Scripts

- Fibonacci sequence (local python vs ray) - [code](./demo_ray_fibonacci.py)
- Prime sequence (local python vs ray) - [code](./demo_ray_primes.py)
- YOLO v8 Inference - [code](./yolov8_inf.py)
- YOLO v8 Batch Inference - [code](./yolov8_batchinf.py)
- YOLO v8 Batch Inference using ray(`task`) - [code](./demo_raytask_yolov8inf.py)
- YOLO v8 Batch Inference using ray(`actor`) - [code](./demo_rayactor_yolov8inf.py)
- A simple ray data demo - [code](./demo_raydata.py)
- A simple ray predict demo which uses pytorch deep learning framework - [code](./demo_raypredict_torch.py)
- A simple ray serve demo - [code](./demo_rayserve_yolov8inf.py) 


## Start Ray AWS cluster

- `cluster.yaml` - configuration file 

- `ray up cluster.yaml`

- `ray submit [script.py]`

- `ray down cluster.yaml`