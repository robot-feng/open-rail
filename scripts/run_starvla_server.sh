#!/usr/bin/env bash
# 启动 RAIL 的 StarVLA 模型服务，加载权重并接收客户端的推理请求。
# 用法（在 /data/open-rail 下执行）：
#   scripts/run_starvla_server.sh data/DP20/checkpoints/steps_25000_pytorch_model.pt
#   scripts/run_starvla_server.sh data/DP20/checkpoints/steps_25000_pytorch_model.pt --debug
# --debug 使用文本日志；省略时使用 run_server.py 默认的终端状态面板。
#
# 模型文件应保留训练时的目录结构：
#   <run_dir>/config.yaml
#   <run_dir>/dataset_statistics.json
#   <run_dir>/checkpoints/<checkpoint>.pt
# StarVLA 适配器读取 framework.name 选择注册算法，并读取 checkpoint 配置与统计文件。
# 相机顺序由 conf/models_conf.py 指定；特殊状态/动作顺序由 checkpoint 的 rail_io 声明。
#
# 推理通信默认值：服务端监听 tcp://*:5566，端口来自 conf/zmq_conf.py。
# 客户端目标在网页配置 vla_zmq.ip / vla_zmq.port 中设置，保存到 conf/default_conf.yaml。
# 同机用 127.0.0.1:5566；异机将客户端的 ip 改成模型服务器实际 IP，双方端口一致。
# run_web_client.py --host / --port 设置网页地址（默认 0.0.0.0:9000），与推理地址分开。
set -euo pipefail

if [[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]]; then
    echo "Usage: $0 CHECKPOINT [--debug]"
    echo "Inference server: tcp://*:5566 by default (conf/zmq_conf.py)."
    echo "Client target: vla_zmq.ip / vla_zmq.port in the web console."
    exit 0
fi

# 先按调用者的当前目录解析权重路径；-e 会在文件不存在时立即退出。
# shift 后剩余的参数原样交给 run_server.py，例如 --debug。
checkpoint="$(realpath -e -- "$1")"
shift
RAIL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 本机默认使用专用的 StarVLA 环境。迁移时可通过环境变量替换这两条路径：
#   STARVLA_ROOT=/path/to/starVLA STARVLA_PYTHON=/path/to/env/bin/python \
#     scripts/run_starvla_server.sh /path/to/run/checkpoints/model.pt
# ml-collections 等依赖需安装在这个 Python 环境中，版本见 pyproject.toml。
STARVLA_ROOT="${STARVLA_ROOT:-/data/EvaClient-StarVLA/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/data/conda-envs/starvla/bin/python}"
env_prefix="$(dirname "$(dirname "${STARVLA_PYTHON}")")"
arch="$(uname -m)"

export STARVLA_ROOT
# 限制 CPU 数值库线程数，减少与相机处理、控制进程争抢 CPU；保留用户的显式设置。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# 查找推理环境附带的原生依赖及 CUDA 库；arch 在本机为 aarch64。
# PYTHONPATH 让该环境能导入 RAIL 和 StarVLA 源码，两项设置均保留调用者已有路径。
export LD_LIBRARY_PATH="${env_prefix}/usr/lib/${arch}-linux-gnu:/usr/local/cuda/targets/${arch}-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${RAIL_ROOT}:${STARVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${RAIL_ROOT}"
# -u 及时输出日志；exec 让 Python 接收 Ctrl+C/SIGTERM，正常释放模型服务资源。
# 此处固定选择 StarVLA 适配器；具体训练算法由权重对应的 config.yaml 决定。
exec "${STARVLA_PYTHON}" -u run_server.py \
    --model_type starvla --model_path "${checkpoint}" \
    "$@"
