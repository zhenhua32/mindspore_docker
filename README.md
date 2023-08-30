[TOC]

# 文档

[安装 mindspore](https://www.mindspore.cn/install)


# 使用镜像

拉取镜像.

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0
```

启动

```bash
docker run -it -v /dev/shm:/dev/shm --runtime=nvidia swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0 /bin/bash

docker run -it --gpus all swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0 /bin/bash
```

验证是否成功

```python
python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
```
