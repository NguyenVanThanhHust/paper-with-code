# papers-implementation
This is where I learn how to implement papers. Some I implemented myself, other I learn from tutorials from others. If I forgot to add origin/refence, pls tell me.

## TODO
https://johanwind.github.io/2022/07/06/dln_classifier.html

## Install 
Build docker image
```
docker build -t dali_img -f ./dockers/dali.Dockerfile ./dockers/
```

Build docker container
```
docker run --name opt4train_ctn -it --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace/ -it dali_img:latest /bin/bash
```