# 这个脚本用来完整产出所有的blender渲染的数据
export blender=/apdcephfs_cq11/share_1467498/softwares/blender-3.6.18-linux-x64/blender
export PYTHONPATH=.

# ===============================================
# 方式1: 使用调度器批量运行 (推荐)
# ===============================================

# 列出所有可用任务
python examples/fig_v2m/render_scheduler.py --config examples/fig_v2m/render_config.json --list

# 运行指定任务 (使用 8 GPU 并行)
python examples/fig_v2m/render_scheduler.py \
    --config examples/fig_v2m/render_config.json \
    --tasks baichuyu_wovideo,FWUCjj44YIg_71,FWUCjj44YIg_71_over \
    --gpus 0,1,2,3,4,5,6,7 \
    --blender ${blender} \
    --num-samples 512

# 运行所有任务
# python examples/fig_v2m/render_scheduler.py \
#     --config examples/fig_v2m/render_config.json \
#     --all \
#     --blender ${blender}

# Dry-run 模式 (只打印命令)
# python examples/fig_v2m/render_scheduler.py \
#     --config examples/fig_v2m/render_config.json \
#     --all \
#     --blender ${blender} \
#     --dry-run

# Debug 模式 (低分辨率快速预览)
# python examples/fig_v2m/render_scheduler.py \
#     --config examples/fig_v2m/render_config.json \
#     --tasks baichuyu_wovideo \
#     --blender ${blender} \
#     --debug

# ===============================================
# 方式2: 直接使用 blender 运行单个任务
# ===============================================

# 使用 JSON 配置
# ${blender} --background --python examples/fig_v2m/render_fbx_video3d.py -- \
#     --name baichuyu_wovideo \
#     --config examples/fig_v2m/render_config.json \
#     --num_samples 256 \
#     --render
