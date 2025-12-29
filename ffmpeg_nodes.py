"""
FFmpeg Video Merge Node - ComfyUI Video Processing Nodes
"""
import os
import subprocess
import tempfile
import uuid
import folder_paths
from .ffmpeg_utils import ensure_ffmpeg

# Try to import ComfyUI's native VIDEO type for compatibility
try:
    from comfy_api.input.video_types import VideoFromFile
    HAS_VIDEO_TYPE = True
except ImportError:
    try:
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        HAS_VIDEO_TYPE = True
    except ImportError:
        HAS_VIDEO_TYPE = False
        VideoFromFile = None


class FFmpeg_LoadVideoFromURL_JX:
    """Load video from URL"""
    
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter video URL, e.g. https://example.com/video.mp4",
                    "dynamicPrompts": False
                }),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    FUNCTION = "load_video_url"
    CATEGORY = "video/ffmpeg"

    def load_video_url(self, url, show_preview=True):
        import requests
        
        if not url or not url.strip():
            return (None,)
        
        url = url.strip()
        
        # Download video to temp file for compatibility with ComfyUI native nodes
        try:
            # Generate temp file path
            ext = os.path.splitext(url.split('?')[0])[-1] or '.mp4'
            if ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                ext = '.mp4'
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_url_video_{uuid.uuid4().hex[:8]}{ext}")
            
            # Download video
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Return VideoFromFile if available, otherwise return path
            if HAS_VIDEO_TYPE and VideoFromFile is not None:
                return (VideoFromFile(temp_path),)
            else:
                return (temp_path,)
                
        except Exception as e:
            print(f"[FFmpeg] Failed to download video from URL: {e}")
            # Fallback to URL string for our own nodes
            return (url,)

class FFmpeg_LoadAudioFromURL_JX:
    """Load audio from URL"""
    
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter audio URL, e.g. https://example.com/audio.mp3",
                    "dynamicPrompts": False
                }),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "load_audio_url"
    CATEGORY = "video/ffmpeg"

    def load_audio_url(self, url, show_preview=True):
        import requests
        import torch
        
        if not url or not url.strip():
            return (None,)
        
        url = url.strip()
        
        # Download audio to temp file
        try:
            # Generate temp file path
            ext = os.path.splitext(url.split('?')[0])[-1] or '.mp3'
            if ext.lower() not in ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a']:
                ext = '.mp3'
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_url_audio_{uuid.uuid4().hex[:8]}{ext}")
            
            # Download audio
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert to WAV using FFmpeg for consistent format
            ffmpeg_path = ensure_ffmpeg()
            wav_path = os.path.join(self.temp_dir, f"ffmpeg_url_audio_{uuid.uuid4().hex[:8]}.wav")
            
            cmd = [
                ffmpeg_path, "-y",
                "-i", temp_path,
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[FFmpeg] FFmpeg conversion failed: {result.stderr}")
                # Try returning path as fallback for our own nodes
                return (temp_path,)
            
            # Load audio using torchaudio
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                
                # Add batch dimension if needed [B, C, S]
                if len(waveform.shape) == 2:
                    waveform = waveform.unsqueeze(0)
                
                # Return ComfyUI standard AUDIO format
                audio_output = {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }
                return (audio_output,)
                
            except ImportError:
                print("[FFmpeg] torchaudio not available, returning file path")
                return (wav_path,)
                
        except Exception as e:
            print(f"[FFmpeg] Failed to download/process audio from URL: {e}")
            # Fallback to URL string for our own nodes
            return (url,)

class FFmpeg_LoadImageFromURL_JX:
    """Load image from URL"""
    
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter image URL",
                    "dynamicPrompts": False
                }),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_image_url"
    CATEGORY = "video/ffmpeg"

    def load_image_url(self, url, show_preview=True):
        import torch
        import numpy as np
        import requests
        from PIL import Image
        from io import BytesIO
        
        if not url or not url.strip():
            # Return placeholder if no URL
            return (torch.zeros((1, 64, 64, 3)),)
        
        try:
            # Download image from URL
            response = requests.get(url.strip(), timeout=30)
            response.raise_for_status()
            
            # Open image with PIL
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode == 'RGBA':
                # Extract alpha channel for mask
                alpha = img.split()[3]
                mask = np.array(alpha).astype(np.float32) / 255.0
                mask = 1.0 - mask  # Invert mask (ComfyUI convention)
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                mask = np.zeros((img.height, img.width), dtype=np.float32)
            else:
                mask = np.zeros((img.height, img.width), dtype=np.float32)
            
            # Convert to numpy array and normalize to 0-1
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Convert to torch tensor with batch dimension [B, H, W, C]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            return (img_tensor,)
            
        except Exception as e:
            return (torch.zeros((1, 64, 64, 3)),)


class FFmpeg_VideoMerge_JX:
    """Use FFmpeg to merge multiple videos and audios"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.temp_files = []  # Track temp files for cleanup
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "audio_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "number"
                }),
                "audio_mode": (["mix", "replace"], {"default": "mix"}),
                "video_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "output_name": ("STRING", {"default": "merged_output"}),
                "save_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_1": ("VIDEO", {}),
                "video_2": ("VIDEO", {}),
                "video_3": ("VIDEO", {}),
                "video_4": ("VIDEO", {}),
                "video_5": ("VIDEO", {}),
                "video_6": ("VIDEO", {}),
                "video_7": ("VIDEO", {}),
                "video_8": ("VIDEO", {}),
                "video_9": ("VIDEO", {}),
                "video_10": ("VIDEO", {}),
                "audio_1": ("AUDIO", {}),
                "audio_2": ("AUDIO", {}),
                "audio_3": ("AUDIO", {}),
                "audio_4": ("AUDIO", {}),
                "audio_5": ("AUDIO", {}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    FUNCTION = "merge_media"
    CATEGORY = "video/ffmpeg"
    OUTPUT_NODE = True

    
    def _extract_video_path(self, video_input, index):
        """Extract video file path from various input formats, save to temp file if needed"""
        if video_input is None:
            return None
            
        # If string, use directly (could be URL or local path)
        if isinstance(video_input, str):
            # Accept URLs directly
            if video_input.startswith(('http://', 'https://')):
                return video_input
            # Accept existing local files
            if os.path.exists(video_input):
                return video_input
            return None
        
        # Check for save_to method (VideoFromFile object)
        if hasattr(video_input, 'save_to'):
            # Save to temp file
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_merge_temp_{index}_{uuid.uuid4().hex[:8]}.mp4")
            try:
                video_input.save_to(temp_path)
                self.temp_files.append(temp_path)
                return temp_path
            except Exception as e:
                pass
        
        # Try to get file path from __dict__
        if hasattr(video_input, '__dict__'):
            for key, value in video_input.__dict__.items():
                if isinstance(value, str) and os.path.exists(value):
                    return value
                # Check if it is a file path string
                if isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower()):
                    if os.path.exists(value):
                        return value
        
        # If dict
        if isinstance(video_input, dict):
            for key in ['video_path', 'path', 'filename', 'file', 'video', 'source']:
                if key in video_input and isinstance(video_input[key], str):
                    if os.path.exists(video_input[key]):
                        return video_input[key]
        
        # If tuple or list
        if isinstance(video_input, (tuple, list)) and len(video_input) > 0:
            return self._extract_video_path(video_input[0], index)
        
        return None
    
    def merge_media(self, video_count, audio_count, audio_mode, video_volume, audio_volume, output_name,
                     save_video=True,
                     video_1=None, video_2=None, video_3=None, video_4=None, video_5=None,
                     video_6=None, video_7=None, video_8=None, video_9=None, video_10=None,
                     audio_1=None, audio_2=None, audio_3=None, audio_4=None, audio_5=None):
        
        self.temp_files = []  # Reset temp file list
        
        
        # Collect video paths
        all_videos = [video_1, video_2, video_3, video_4, video_5,
                      video_6, video_7, video_8, video_9, video_10]
        videos = []
        
        for i in range(video_count):
            video_input = all_videos[i]
            if video_input is None:
                continue
            video_path = self._extract_video_path(video_input, i + 1)
            if video_path:
                # Accept both URLs and local files
                if video_path.startswith(('http://', 'https://')) or os.path.exists(video_path):
                    videos.append(video_path)
        
        # Collect audio paths
        all_audios = [audio_1, audio_2, audio_3, audio_4, audio_5]
        audios = []
        
        for i in range(audio_count):
            audio_input = all_audios[i]
            if audio_input is None:
                continue
            audio_path = self._extract_audio_path(audio_input, i + 1)
            if audio_path:
                # Accept both URLs and local files
                if audio_path.startswith(('http://', 'https://')) or os.path.exists(audio_path):
                    audios.append(audio_path)
        
        # Validate inputs - need at least 2 media files total
        total_inputs = len(videos) + len(audios)
        if total_inputs < 2:
            self._cleanup_temp_files()
            raise ValueError(f"Need at least 2 media files, currently only {total_inputs}")
        
        # Ensure FFmpeg available
        ffmpeg_path = ensure_ffmpeg()
        
        # Create output path with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_name}_{timestamp}.mp4"
        
        if save_video:
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            output_path = os.path.join(self.temp_dir, f"{output_name}_{uuid.uuid4().hex[:8]}.mp4")
        
        # Build and execute FFmpeg command based on input types
        if len(videos) >= 2 and len(audios) == 0:
            # Multiple videos only - concatenate
            self._merge_videos_only(ffmpeg_path, videos, output_path)
        elif len(videos) == 0 and len(audios) >= 2:
            # Multiple audios only - concatenate  
            output_filename = f"{output_name}_{timestamp}.mp3"
            if save_video:
                output_path = os.path.join(self.output_dir, output_filename)
            else:
                output_path = os.path.join(self.temp_dir, f"{output_name}_{uuid.uuid4().hex[:8]}.mp3")
            self._merge_audios_only(ffmpeg_path, audios, output_path)
        elif len(videos) == 1 and len(audios) >= 1:
            # Single video + audio(s) - add audio to video
            self._merge_video_with_audios(ffmpeg_path, videos[0], audios, output_path, audio_mode, video_volume, audio_volume)
        elif len(videos) >= 2 and len(audios) >= 1:
            # Multiple videos + audio(s) - concat videos then add audio
            self._merge_videos_and_audios(ffmpeg_path, videos, audios, output_path, audio_mode, video_volume, audio_volume)
        elif len(videos) >= 1 and len(audios) >= 1:
            # 1 video + 1 or more audio
            self._merge_video_with_audios(ffmpeg_path, videos[0], audios, output_path, audio_mode, video_volume, audio_volume)
        else:
            self._cleanup_temp_files()
            raise ValueError("Invalid input combination")
        
        self._cleanup_temp_files()
        
        # Create VIDEO output - use VideoFromFile if available, otherwise return path string
        if HAS_VIDEO_TYPE and VideoFromFile is not None:
            video_output = VideoFromFile(output_path)
        else:
            video_output = output_path
        
        # Return with proper UI format
        # Always include preview data, but only add to Media Assets when save_video is True
        ui_data = {}
        if save_video:
            # Show in Media Assets panel
            ui_data["videos"] = [{"filename": output_filename, "subfolder": "", "type": "output"}]
        # Always include preview path for node preview (using temp type for temp files)
        preview_filename = output_filename if save_video else os.path.basename(output_path)
        preview_type = "output" if save_video else "temp"
        ui_data["VIDEO"] = [preview_filename]  # For node preview
        
        return {"ui": ui_data, "result": (video_output,)}
    
    def _extract_audio_path(self, audio_input, index):
        """Extract audio file path from various input formats"""
        if audio_input is None:
            return None
        
        # If string, use directly (could be URL or local path)
        if isinstance(audio_input, str):
            # Accept URLs directly
            if audio_input.startswith(('http://', 'https://')):
                return audio_input
            # Accept existing local files
            if os.path.exists(audio_input):
                return audio_input
            return None
        
        # Check for dict with waveform (ComfyUI AUDIO type)
        if isinstance(audio_input, dict) and 'waveform' in audio_input:
            return self._save_audio_tensor(audio_input, index)
        
        # Try from __dict__
        if hasattr(audio_input, '__dict__'):
            for key, value in audio_input.__dict__.items():
                if isinstance(value, str) and os.path.exists(value):
                    return value
        
        return None
    
    def _save_audio_tensor(self, audio_dict, index):
        """Save ComfyUI AUDIO type to file"""
        try:
            import torchaudio
            waveform = audio_dict.get('waveform')
            sample_rate = audio_dict.get('sample_rate', 44100)
            if waveform is None:
                return None
            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_audio_{index}_{uuid.uuid4().hex[:8]}.wav")
            torchaudio.save(temp_path, waveform, sample_rate)
            self.temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            return None
    
    def _merge_videos_only(self, ffmpeg_path, videos, output_path):
        """Merge multiple videos by concatenation"""
        
        # Check if any video is a URL - concat demuxer doesn't support URLs
        has_url = any(v.startswith(('http://', 'https://')) for v in videos)
        
        if has_url:
            # Use filter_complex method for URLs
            cmd = self._build_filter_command(ffmpeg_path, videos, output_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            return
        
        # For local files only, try concat demuxer first (faster, stream copy)
        list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        try:
            for video in videos:
                abs_path = os.path.abspath(video).replace("\\", "/")
                list_file.write(f"file '{abs_path}'\n")
            list_file.close()
            
            cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Try re-encoding with filter_complex
                cmd = self._build_filter_command(ffmpeg_path, videos, output_path)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
        finally:
            os.unlink(list_file.name)
    
    def _merge_audios_only(self, ffmpeg_path, audios, output_path):
        """Merge multiple audios by concatenation"""
        list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        try:
            for audio in audios:
                abs_path = os.path.abspath(audio).replace("\\", "/")
                list_file.write(f"file '{abs_path}'\n")
            list_file.close()
            
            cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
        finally:
            os.unlink(list_file.name)
    
    def _merge_video_with_audios(self, ffmpeg_path, video_path, audios, output_path, audio_mode="mix", video_volume=1.0, audio_volume=1.0):
        """Add audio track(s) to a single video.
        
        Args:
            audio_mode: 'replace' to discard original video audio, 'mix' to mix with original
            video_volume: Volume multiplier for original video audio (0.0-2.0)
            audio_volume: Volume multiplier for external audio (0.0-2.0)
        """
        cmd = [ffmpeg_path, "-y"]
        
        # Add video input
        cmd.extend(["-i", video_path])
        
        # Add audio inputs
        for audio in audios:
            cmd.extend(["-i", audio])
        
        # Build filter based on mode
        if audio_mode == "replace":
            # Replace mode: discard original video audio, use external audio(s) only
            if len(audios) == 1:
                # Single external audio with volume control
                filter_str = f"[1:a]volume={audio_volume}[aout]"
                cmd.extend(["-filter_complex", filter_str, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
            else:
                # Multiple external audios - concat them (串联而非叠加)
                filter_parts = []
                for i in range(len(audios)):
                    filter_parts.append(f"[{i+1}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={audio_volume}[ea{i}];")
                ea_streams = "".join([f"[ea{i}]" for i in range(len(audios))])
                filter_str = "".join(filter_parts) + f"{ea_streams}concat=n={len(audios)}:v=0:a=1[aout]"
                cmd.extend(["-filter_complex", filter_str, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
        else:
            # Mix mode: mix original video audio with external audio(s)
            if len(audios) == 1:
                # Mix original video audio with single external audio
                filter_str = f"[0:a]volume={video_volume}[va];[1:a]volume={audio_volume}[ea];[va][ea]amix=inputs=2:duration=first[aout]"
                cmd.extend(["-filter_complex", filter_str, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
            else:
                # Multiple external audios - concat them first, then mix with video audio
                filter_parts = [f"[0:a]volume={video_volume}[va];"]
                for i in range(len(audios)):
                    filter_parts.append(f"[{i+1}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={audio_volume}[ea{i}];")
                ea_streams = "".join([f"[ea{i}]" for i in range(len(audios))])
                filter_str = "".join(filter_parts) + f"{ea_streams}concat=n={len(audios)}:v=0:a=1[concatea];[va][concatea]amix=inputs=2:duration=first[aout]"
                cmd.extend(["-filter_complex", filter_str, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
    
    def _merge_videos_and_audios(self, ffmpeg_path, videos, audios, output_path, audio_mode="mix", video_volume=1.0, audio_volume=1.0):
        """Merge videos together and overlay audio tracks.
        
        Uses a single FFmpeg command with filter_complex for reliability,
        especially when handling mixed local files and URLs.
        
        Args:
            audio_mode: 'replace' to discard original video audio, 'mix' to mix with original
            video_volume: Volume multiplier for original video audio (0.0-2.0)
            audio_volume: Volume multiplier for external audio (0.0-2.0)
        """
        
        # Build a single FFmpeg command to concat videos and mix audios
        cmd = [ffmpeg_path, "-y"]
        
        # Add all video inputs
        for video in videos:
            cmd.extend(["-i", video])
        
        # Add all audio inputs
        for audio in audios:
            cmd.extend(["-i", audio])
        
        n_videos = len(videos)
        n_audios = len(audios)
        
        # Normalize resolution, fps, and pixel format
        target_w = 1080
        target_h = 1920
        target_fps = 30
        
        filter_parts = []
        
        # Process each video: scale, pad, fps, format
        for i in range(n_videos):
            filter_parts.append(
                f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"fps={target_fps},format=yuv420p,setsar=1[v{i}];"
            )
            # Normalize video's original audio with volume control (only needed for mix mode)
            if audio_mode == "mix":
                filter_parts.append(
                    f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={video_volume}[va{i}];"
                )
        
        if audio_mode == "mix":
            # Mix mode: concat videos with their audio, then concat external audios and mix
            
            # Concat all videos (interleaved: [v0][a0][v1][a1]...)
            interleaved = "".join([f"[v{i}][va{i}]" for i in range(n_videos)])
            filter_parts.append(f"{interleaved}concat=n={n_videos}:v=1:a=1[concatv][concata];")
            
            # Normalize external audio inputs with volume control
            for i in range(n_audios):
                audio_idx = n_videos + i  # Audio inputs come after video inputs
                filter_parts.append(
                    f"[{audio_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={audio_volume}[ea{i}];"
                )
            
            # Concat all external audios (串联而非叠加)
            if n_audios == 1:
                # Single external audio - mix with video audio
                filter_parts.append("[concata][ea0]amix=inputs=2:duration=first[aout]")
            else:
                # Multiple external audios - concat them first, then mix with video audio
                ea_streams = "".join([f"[ea{i}]" for i in range(n_audios)])
                filter_parts.append(f"{ea_streams}concat=n={n_audios}:v=0:a=1[concatea];")
                filter_parts.append("[concata][concatea]amix=inputs=2:duration=first[aout]")
        else:
            # Replace mode: concat videos (video only), use only external audios (concatenated)
            
            # Concat all video streams only
            video_streams = "".join([f"[v{i}]" for i in range(n_videos)])
            filter_parts.append(f"{video_streams}concat=n={n_videos}:v=1:a=0[concatv];")
            
            # Normalize external audio inputs with volume control
            for i in range(n_audios):
                audio_idx = n_videos + i
                filter_parts.append(
                    f"[{audio_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={audio_volume}[ea{i}];"
                )
            
            # Concat all external audios (串联而非叠加)
            if n_audios == 1:
                filter_parts.append("[ea0]anull[aout]")
            else:
                ea_streams = "".join([f"[ea{i}]" for i in range(n_audios)])
                filter_parts.append(f"{ea_streams}concat=n={n_audios}:v=0:a=1[aout]")
        
        filter_str = "".join(filter_parts)
        
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[concatv]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "ultrafast",  # 最快编码速度
            "-crf", "23",  # 质量控制（越低质量越好，但速度越慢）
            "-threads", "0",  # 使用所有可用CPU核心
            "-c:a", "aac",
            "-b:a", "128k",  # 降低音频比特率提高速度
            output_path
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
    
    def _cleanup_temp_files(self):
        """Clean up temp files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files = []
    
    def _build_filter_command(self, ffmpeg_path, videos, output_path):
        """Build re-encode command with resolution normalization.
        
        Scales all videos to a common resolution (1080x1920 for portrait, 1920x1080 for landscape)
        and normalizes FPS, pixel format, and audio to ensure successful concatenation.
        """
        cmd = [ffmpeg_path, "-y"]
        
        for video in videos:
            cmd.extend(["-i", video])
        
        n = len(videos)
        
        # Build filter: scale, pad, set fps and pixel format for each video
        # Use 1080x1920 (9:16 portrait) as default target - common for vertical videos
        # If videos are landscape, use 1920x1080
        target_w = 1080
        target_h = 1920
        target_fps = 30
        
        filter_parts = []
        
        # For each video: scale to fit, pad to exact size, set fps and pixel format
        # Also handle audio: if no audio, create silent audio
        for i in range(n):
            # Video processing: scale to fit within target, pad to exact target size
            filter_parts.append(
                f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"fps={target_fps},format=yuv420p,setsar=1[v{i}];"
            )
            # Audio processing: normalize and handle missing audio
            filter_parts.append(
                f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}];"
            )
        
        # Concat all normalized streams
        # Concat expects alternating streams: [v0][a0][v1][a1]...
        interleaved = "".join([f"[v{i}][a{i}]" for i in range(n)])
        filter_parts.append(f"{interleaved}concat=n={n}:v=1:a=1[outv][outa]")
        
        filter_str = "".join(filter_parts)
        
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ])
        
        return cmd


class FFmpeg_VideoAudioMerge_JX:
    """Use FFmpeg to merge audio into video"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.temp_files = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("VIDEO", {}),
                "audio_path": ("AUDIO", {}),
                "output_name": ("STRING", {"default": "video_with_audio"}),
                "audio_mode": (["replace", "mix"], {"default": "replace"}),
                "video_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "audio_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "loop_audio": ("BOOLEAN", {"default": True}),
                "save_video": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    FUNCTION = "merge_video_audio"
    CATEGORY = "video/ffmpeg"
    OUTPUT_NODE = True
    
    def _save_audio_tensor(self, audio_dict, index):
        """Save ComfyUI AUDIO type (waveform tensor) as audio file"""
        try:
            import torchaudio
            
            waveform = audio_dict.get('waveform')
            sample_rate = audio_dict.get('sample_rate', 44100)
            
            if waveform is None:
                return None
            
            # Ensure waveform is correct format
            if len(waveform.shape) == 3:
                waveform = waveform.squeeze(0)
            
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_audio_{index}_{uuid.uuid4().hex[:8]}.wav")
            torchaudio.save(temp_path, waveform, sample_rate)
            self.temp_files.append(temp_path)
            return temp_path
        except Exception as e:
            return None
    
    def _extract_media_path(self, media_input, media_type, index=1):
        """Extract media file path from various input formats"""
        if media_input is None:
            return None
        
        
        # If string, use directly (local path or URL)
        if isinstance(media_input, str):
            # Accept URLs directly
            if media_input.startswith(('http://', 'https://')):
                return media_input
            # Accept existing local files
            if os.path.exists(media_input):
                return media_input
            return None
        
        # Check for save_to method (VideoFromFile object)
        if hasattr(media_input, 'save_to'):
            ext = ".mp4" if media_type == "video" else ".wav"
            temp_path = os.path.join(self.temp_dir, f"ffmpeg_merge_{media_type}_{index}_{uuid.uuid4().hex[:8]}{ext}")
            try:
                media_input.save_to(temp_path)
                self.temp_files.append(temp_path)
                return temp_path
            except Exception as e:
                pass
        
        # If dict - check if is ComfyUI AUDIO type
        if isinstance(media_input, dict):
            # Check if contains waveform (ComfyUI AUDIO type)
            if 'waveform' in media_input:
                return self._save_audio_tensor(media_input, index)
            
            # try to get file from dictpath
            for key in ['path', 'filename', 'file', 'video_path', 'audio_path', 'source']:
                if key in media_input and isinstance(media_input[key], str):
                    if os.path.exists(media_input[key]):
                        return media_input[key]
            
            # print dict contents for debugging
        
        # Try to get file path from __dict__
        if hasattr(media_input, '__dict__'):
            for key, value in media_input.__dict__.items():
                if isinstance(value, str) and os.path.exists(value):
                    return value
        
        # If tuple or list
        if isinstance(media_input, (tuple, list)) and len(media_input) > 0:
            return self._extract_media_path(media_input[0], media_type, index)
        
        return None
    
    def merge_video_audio(self, video_path, audio_path,
                          output_name, audio_mode,
                          video_volume=1.0, audio_volume=1.0,
                          loop_audio=True, save_video=True):
        
        self.temp_files = []
        
        # Extract actual path from VideoFromFile or other types
        video_path = self._extract_media_path(video_path, "video", 1)
        audio_path = self._extract_media_path(audio_path, "audio", 1)
        
        
        # Validate paths (URLs are accepted directly, local files must exist)
        def is_valid_path(path):
            if not path:
                return False
            if path.startswith(('http://', 'https://')):
                return True
            return os.path.exists(path)
        
        if not is_valid_path(video_path):
            raise ValueError(f"video file not found: {video_path}")
        if not is_valid_path(audio_path):
            raise ValueError(f"audio file not found: {audio_path}")
        
        # ensure FFmpeg available
        ffmpeg_path = ensure_ffmpeg()
        
        # Determine output path based on save_video setting
        # Add timestamp to filename to avoid overwriting
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{output_name}_{timestamp}.mp4"
        if save_video:
            # Save to output directory (permanent)
            output_path = os.path.join(self.output_dir, output_filename)
        else:
            # Save to temp directory (will be cleaned up)
            output_path = os.path.join(self.temp_dir, f"{output_name}_{uuid.uuid4().hex[:8]}.mp4")
        
        try:
            # merge video and audio
            self._merge_video_with_audio(ffmpeg_path, video_path, audio_path,
                                         output_path, audio_mode,
                                         video_volume, audio_volume,
                                         loop_audio)
            
        finally:
            self._cleanup_temp_files()
        
        # Create VIDEO output - use VideoFromFile if available, otherwise return path string
        if HAS_VIDEO_TYPE and VideoFromFile is not None:
            video_output = VideoFromFile(output_path)
        else:
            video_output = output_path
        
        # Return with proper UI format
        # Always include preview data, but only add to Media Assets when save_video is True
        ui_data = {}
        if save_video:
            # Show in Media Assets panel
            ui_data["videos"] = [{"filename": output_filename, "subfolder": "", "type": "output"}]
        # Always include preview path for node preview (using temp type for temp files)
        preview_filename = output_filename if save_video else os.path.basename(output_path)
        preview_type = "output" if save_video else "temp"
        ui_data["VIDEO"] = [preview_filename]  # For node preview
        
        return {"ui": ui_data, "result": (video_output,)}
    
    def _merge_videos(self, ffmpeg_path, video_paths, output_path):
        """mergemultiple video"""
        
        list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        try:
            for video in video_paths:
                abs_path = os.path.abspath(video).replace("\\", "/")
                list_file.write(f"file '{abs_path}'\n")
            list_file.close()
            
            cmd = [
                ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file.name,
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # try re-encoding
                cmd = [ffmpeg_path, "-y"]
                for video in video_paths:
                    cmd.extend(["-i", video])
                
                n = len(video_paths)
                filter_parts = []
                for i in range(n):
                    filter_parts.append(f"[{i}:v:0]")
                filter_str = "".join(filter_parts) + f"concat=n={n}:v=1:a=0[outv]"
                
                cmd.extend([
                    "-filter_complex", filter_str,
                    "-map", "[outv]",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-an",
                    output_path
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"videoMerge failed: {result.stderr}")
        finally:
            os.unlink(list_file.name)
    
    def _merge_audios(self, ffmpeg_path, audio_paths, output_path):
        """mergemultiple audio（concat）"""
        
        list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        try:
            for audio in audio_paths:
                abs_path = os.path.abspath(audio).replace("\\", "/")
                list_file.write(f"file '{abs_path}'\n")
            list_file.close()
            
            cmd = [
                ffmpeg_path,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file.name,
                "-c:a", "pcm_s16le",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # tryusing filter_complex merge
                cmd = [ffmpeg_path, "-y"]
                for audio in audio_paths:
                    cmd.extend(["-i", audio])
                
                n = len(audio_paths)
                filter_parts = []
                for i in range(n):
                    filter_parts.append(f"[{i}:a]")
                filter_str = "".join(filter_parts) + f"concat=n={n}:v=0:a=1[outa]"
                
                cmd.extend([
                    "-filter_complex", filter_str,
                    "-map", "[outa]",
                    "-c:a", "pcm_s16le",
                    output_path
                ])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"audioMerge failed: {result.stderr}")
        finally:
            os.unlink(list_file.name)
    
    def _merge_video_with_audio(self, ffmpeg_path, video_path, audio_path, 
                                 output_path, audio_mode,
                                 video_volume, audio_volume, 
                                 loop_audio):
        """Merge audio into video. Output duration always matches video duration."""
        
        if audio_mode == "replace":
            # Replace mode: replace original video audio with new audio
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", video_path,
            ]
            
            if loop_audio:
                # Loop audio indefinitely, then use -shortest to trim to video length
                cmd.extend(["-stream_loop", "-1", "-i", audio_path])
                # Simple volume filter when looping
                filter_complex = f"[1:a]volume={audio_volume}[aout]"
                cmd.extend([
                    "-filter_complex", filter_complex,
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",  # Use video duration as reference
                    output_path
                ])
            else:
                # No loop: pad audio with silence to match video duration
                # apad will extend audio with silence if it's shorter than video
                filter_complex = f"[1:a]volume={audio_volume},apad[aout]"
                cmd.extend(["-i", audio_path])
                cmd.extend([
                    "-filter_complex", filter_complex,
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",  # This will now use video duration since audio is padded infinitely
                    output_path
                ])
            
        else:
            # mix mode: mix new audio with original video audio
            # First, check if video has audio track
            probe_cmd = [
                ffmpeg_path,
                "-i", video_path,
                "-hide_banner"
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            has_audio = "Audio:" in probe_result.stderr
            
            if has_audio:
                # Video has audio - mix them
                # duration=first means output matches first input (video audio) duration
                filter_complex = f"[0:a]volume={video_volume}[v_audio];[1:a]volume={audio_volume}[a_audio];[v_audio][a_audio]amix=inputs=2:duration=first[aout]"
            else:
                # Video has no audio - just use external audio
                filter_complex = f"[1:a]volume={audio_volume}[aout]"
            
            cmd = [
                ffmpeg_path,
                "-y",
                "-i", video_path,
            ]
            
            # -stream_loop must come BEFORE the input it applies to
            if loop_audio:
                cmd.extend(["-stream_loop", "-1", "-i", audio_path])
            else:
                cmd.extend(["-i", audio_path])
            
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",  # Always use video duration as reference
            ])
            
            cmd.append(output_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
    
    def _cleanup_temp_files(self):
        """Clean up temp files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        self.temp_files = []


NODE_CLASS_MAPPINGS = {
    "FFmpeg_LoadVideoFromURL_JX": FFmpeg_LoadVideoFromURL_JX,
    "FFmpeg_LoadAudioFromURL_JX": FFmpeg_LoadAudioFromURL_JX,
    "FFmpeg_LoadImageFromURL_JX": FFmpeg_LoadImageFromURL_JX,
    "FFmpeg_VideoMerge_JX": FFmpeg_VideoMerge_JX,
    "FFmpeg_VideoAudioMerge_JX": FFmpeg_VideoAudioMerge_JX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FFmpeg_LoadVideoFromURL_JX": "FFmpeg Video URL",
    "FFmpeg_LoadAudioFromURL_JX": "FFmpeg Audio URL",
    "FFmpeg_LoadImageFromURL_JX": "FFmpeg Image URL",
    "FFmpeg_VideoMerge_JX": "FFmpeg Media Merge",
    "FFmpeg_VideoAudioMerge_JX": "FFmpeg Single AV Merge"
}

