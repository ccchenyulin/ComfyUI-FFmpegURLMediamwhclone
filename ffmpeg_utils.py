"""
FFmpeg Utilities - 自动下载和管理 FFmpeg
"""
import os
import sys
import zipfile
import urllib.request
import shutil
import time

def get_ffmpeg_path():
    """获取 FFmpeg 可执行文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(current_dir, "bin")
    
    if sys.platform == "win32":
        ffmpeg_path = os.path.join(bin_dir, "ffmpeg.exe")
    else:
        ffmpeg_path = os.path.join(bin_dir, "ffmpeg")
    
    return ffmpeg_path

def is_ffmpeg_available():
    """检查 FFmpeg 是否已下载"""
    ffmpeg_path = get_ffmpeg_path()
    return os.path.exists(ffmpeg_path)

def download_ffmpeg():
    """下载 FFmpeg"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(current_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    if sys.platform != "win32":
        raise RuntimeError("自动下载仅支持 Windows，请手动安装 FFmpeg")
    
    print("[FFmpeg Video Merge] 正在下载 FFmpeg，请稍候...")
    
    # 使用 gyan.dev 的 essentials 版本（较小）
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = os.path.join(bin_dir, "ffmpeg.zip")
    target_path = os.path.join(bin_dir, "ffmpeg.exe")
    
    # 如果已存在旧的zip文件，先尝试删除
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except:
            # 如果删除失败，使用新文件名
            zip_path = os.path.join(bin_dir, f"ffmpeg_{int(time.time())}.zip")
    
    try:
        # 下载文件
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                print(f"\r[FFmpeg Video Merge] 下载进度: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, zip_path, reporthook=report_progress)
        print()  # 换行
        
        print("[FFmpeg Video Merge] 正在解压...")
        
        # 解压并找到 ffmpeg.exe
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.namelist():
                if file_info.endswith("bin/ffmpeg.exe"):
                    # 读取文件内容
                    data = zip_ref.read(file_info)
                    # 写入目标文件
                    with open(target_path, 'wb') as target:
                        target.write(data)
                    print(f"[FFmpeg Video Merge] 已提取: {target_path}")
                    break
        
        # 等待一下确保文件句柄释放
        time.sleep(0.5)
        
        # 清理 zip 文件
        try:
            os.remove(zip_path)
        except Exception as e:
            print(f"[FFmpeg Video Merge] 警告: 无法删除临时文件 {zip_path}: {e}")
            # 不抛出异常，因为ffmpeg已经解压成功
        
        print("[FFmpeg Video Merge] FFmpeg 下载完成！")
        return True
        
    except Exception as e:
        print(f"[FFmpeg Video Merge] 下载失败: {e}")
        # 尝试清理
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except:
            pass
        raise RuntimeError(f"FFmpeg 下载失败: {e}")

def ensure_ffmpeg():
    """确保 FFmpeg 可用，不可用则下载"""
    if not is_ffmpeg_available():
        download_ffmpeg()
    return get_ffmpeg_path()
