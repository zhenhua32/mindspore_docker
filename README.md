[TOC]

原本想直接在 windows 上运行的, 但是不支持 GPU. 只能用了 docker 了.
windows 只能用 CPU 也太惨了点.

# 文档

[安装 mindspore](https://www.mindspore.cn/install)


# 使用镜像

拉取镜像. 这个镜像是 python3.7.5 的.

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0
```

启动

```bash
docker run -it -v /dev/shm:/dev/shm --runtime=nvidia swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0 /bin/bash

docker run -it --gpus all swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0 /bin/bash

docker run -it --gpus all --network host swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-gpu-cuda11.1:2.1.0 /bin/bash
```

验证是否成功

```python
python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
```

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="GPU")

x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

# 目录

原本只是用来学习 mindspore 的, 但现在 mindspore 也不学了. 用来搞别的事情吧.

创建一个 llm 目录, 用来保存各大模型的代码, 先读读它们的代码.




