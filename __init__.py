"""
ComfyUI-FFmpeg-VideoMerge
========================
使用 FFmpeg 合并多个视频的 ComfyUI 节点

功能:
- 支持合并 2-10 个视频
- 首次使用自动下载 FFmpeg
- 输出 MP4 格式
- 节点内预览合并后的视频
"""

print("[FFmpeg Plugin] ========== 开始加载 FFmpeg 插件 ==========")

try:
    from .ffmpeg_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"[FFmpeg Plugin] 成功导入 ffmpeg_nodes 模块")
    print(f"[FFmpeg Plugin] NODE_CLASS_MAPPINGS 包含 {len(NODE_CLASS_MAPPINGS)} 个节点:")
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        print(f"[FFmpeg Plugin]   - {node_name}: {node_class.__name__}")
    print(f"[FFmpeg Plugin] NODE_DISPLAY_NAME_MAPPINGS 包含 {len(NODE_DISPLAY_NAME_MAPPINGS)} 个映射:")
    for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"[FFmpeg Plugin]   - {node_name} -> {display_name}")
except Exception as e:
    import traceback
    print(f"[FFmpeg Plugin] 导入 ffmpeg_nodes 时发生错误: {e}")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

print("[FFmpeg Plugin] ========== FFmpeg 插件加载完成 ==========")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./js"
