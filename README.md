# OpenVINO_Human_Face_detection
Using **face_detection** and **facenet** for recognizing human identities with the OpenVINO library on an Intel CPU.

## Getting Started

### Pre-requirements
Install **docker** before installing the docker container.

- [Tutorial-docker](https://docs.docker.com/engine/install/ubuntu/)

- **Add docker to sudo group** 
    - [Tutourial](https://docs.docker.com/engine/install/linux-postinstall/)
    ``` 
    sudo groupadd docker
    sudo usermod -aG docker $USER
    sudo chmod 777 /var/run/docker.sock
    ```

### Build docker images

```shell
sudo chmod 777 ./docker
sudo ./docker/build.sh -m
```

### Run container

```shell
sudo ./docker/run.sh
```

### Setting config
- [config path](config/config.json)
```json
{
    "det_model":"models/face_detection/version-RFB-320.xml",
    "landmark_model":"models/facenet/facenet_keras.xml",
    "source":"data/feature/Tom_Hiddleston.jpg",
    "init_folder_path":"data/original/",
    "loop":true,
    "device":"CPU",
    "threshold":0.5
}
```

### Initial face features
```python
python3 init_features.py -c config/config.json
```

### Detection results
```python
python3 facial_recognition.py -c config/config.json
```

### Display
<details>
    <summary> Show inference result
    </summary>
      <div align="center">
        <img width="80%" height="80%" src="data/">
      </div>
</details>


## Reference
- Ultra-Light-Fast-Generic-Face-Detector-1MB
    - https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
- Keras-facenet
    - https://github.com/nyoki-mtl/keras-facenet
- Facenet
    - https://github.com/davidsandberg/facenet
- OpenVINO:
    - https://docs.openvino.ai/latest/home.html
    - https://github.com/openvinotoolkit/openvino
