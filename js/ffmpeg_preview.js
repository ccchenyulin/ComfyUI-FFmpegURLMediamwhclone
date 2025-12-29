import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Antigravity.FFmpegNodes_JX",

    async nodeCreated(node) {
        // Only handle our own FFmpeg_*_JX nodes
        if (!node.comfyClass || !node.comfyClass.startsWith("FFmpeg_") || !node.comfyClass.endsWith("_JX")) {
            return;
        }

        // Helper: Get minimum height for node
        const getMinHeight = () => {
            const computed = node.computeSize();
            return computed ? computed[1] : 100;
        };

        // Helper: Enforce size constraint
        const enforceSize = () => {
            const minH = getMinHeight();
            if (node.size[1] > minH + 10) {
                node.setSize([node.size[0], minH]);
            }
        };

        // 1. Multi-Video Merge node
        if (node.comfyClass === "FFmpeg_VideoMerge_JX") {
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                if (originalOnExecuted) originalOnExecuted.call(this, message);

                // Support both old format (VIDEO) and new format (videos)
                let videoPath = null;
                if (message?.VIDEO && message.VIDEO[0]) {
                    videoPath = message.VIDEO[0];
                } else if (message?.videos && message.videos[0]) {
                    videoPath = message.videos[0].filename || message.videos[0];
                }

                if (videoPath) {
                    console.log("[FFmpeg Preview] Video output:", videoPath);

                    // Remove existing preview widget
                    if (this._ffmpegVideoWidget) {
                        if (this._ffmpegVideoWidget.element) this._ffmpegVideoWidget.element.remove();
                        const index = this.widgets.indexOf(this._ffmpegVideoWidget);
                        if (index > -1) this.widgets.splice(index, 1);
                        this._ffmpegVideoWidget = null;
                    }

                    const container = document.createElement("div");
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;overflow:hidden;";

                    const video = document.createElement("video");
                    video.controls = true;
                    video.style.cssText = "width:100%;height:auto;display:block;";

                    // Try output directory first, then temp directory on error
                    const tryLoadVideo = (type) => {
                        video.src = `/view?filename=${encodeURIComponent(videoPath)}&type=${type}&t=${Date.now()}`;
                    };

                    let triedTemp = false;
                    video.onerror = () => {
                        if (!triedTemp) {
                            // Try temp directory if output failed
                            triedTemp = true;
                            console.log("[FFmpeg Preview] Output failed, trying temp directory");
                            tryLoadVideo("temp");
                        } else {
                            console.log("[FFmpeg Preview] Error loading video from both directories");
                            container.innerHTML = '<div style="color:#ff6;padding:10px;text-align:center;">Video preview unavailable</div>';
                            this.setSize([this.size[0], this.computeSize()[1]]);
                        }
                    };

                    video.onloadedmetadata = () => {
                        console.log("[FFmpeg Preview] Video loaded:", video.videoWidth, "x", video.videoHeight);
                        const aspectRatio = video.videoHeight / video.videoWidth;
                        const nodeWidth = Math.max(300, this.size[0]);
                        const previewHeight = nodeWidth * aspectRatio;
                        this.setSize([nodeWidth, this.computeSize()[1] - 50 + previewHeight]);
                    };

                    // Start with output directory
                    tryLoadVideo("output");

                    container.appendChild(video);
                    this._ffmpegVideoWidget = this.addDOMWidget("ffmpeg_preview", "div", container, { serialize: false });
                    this.setSize([this.size[0], this.computeSize()[1]]);
                }
            };

            // Dynamic video and audio inputs
            const videoCountWidget = node.widgets?.find(w => w.name === "video_count");
            const audioCountWidget = node.widgets?.find(w => w.name === "audio_count");

            const updateInputs = () => {
                const targetVideoCount = videoCountWidget?.value || 0;
                const targetAudioCount = audioCountWidget?.value || 0;

                if (!node.inputs) node.inputs = [];

                console.log("[FFmpeg] Target - video:", targetVideoCount, "audio:", targetAudioCount);

                // Step 1: Ensure all required video inputs exist (video_1 to video_N)
                for (let i = 1; i <= targetVideoCount; i++) {
                    const inputName = `video_${i}`;
                    const exists = node.inputs.some(inp => inp.name === inputName);
                    if (!exists) {
                        node.addInput(inputName, "VIDEO");
                        console.log("[FFmpeg] Added " + inputName);
                    }
                }

                // Step 2: Remove video inputs that exceed the target count
                for (let i = 10; i > targetVideoCount; i--) {
                    const inputName = `video_${i}`;
                    const idx = node.inputs.findIndex(inp => inp.name === inputName);
                    if (idx !== -1) {
                        node.removeInput(idx);
                        console.log("[FFmpeg] Removed " + inputName);
                    }
                }

                // Step 3: Ensure all required audio inputs exist (audio_1 to audio_N)
                for (let i = 1; i <= targetAudioCount; i++) {
                    const inputName = `audio_${i}`;
                    const exists = node.inputs.some(inp => inp.name === inputName);
                    if (!exists) {
                        node.addInput(inputName, "AUDIO");
                        console.log("[FFmpeg] Added " + inputName);
                    }
                }

                // Step 4: Remove audio inputs that exceed the target count
                for (let i = 5; i > targetAudioCount; i--) {
                    const inputName = `audio_${i}`;
                    const idx = node.inputs.findIndex(inp => inp.name === inputName);
                    if (idx !== -1) {
                        node.removeInput(idx);
                        console.log("[FFmpeg] Removed " + inputName);
                    }
                }

                // Step 5: Sort inputs so videos come before audios
                if (node.inputs.length > 0) {
                    node.inputs.sort((a, b) => {
                        const aIsVideo = a.name.startsWith("video_");
                        const bIsVideo = b.name.startsWith("video_");
                        if (aIsVideo && !bIsVideo) return -1;
                        if (!aIsVideo && bIsVideo) return 1;
                        // Within same type, sort by number
                        const aNum = parseInt(a.name.split("_")[1]) || 0;
                        const bNum = parseInt(b.name.split("_")[1]) || 0;
                        return aNum - bNum;
                    });
                }

                // Only resize if no preview widget exists, otherwise preserve current height
                const hasPreview = node._ffmpegVideoWidget || node.widgets?.some(w => w.name === "ffmpeg_preview");
                if (!hasPreview) {
                    node.setSize([node.size[0], node.computeSize()[1]]);
                }
                app.graph.setDirtyCanvas(true);
            };

            // Hook into widget callbacks
            if (videoCountWidget) {
                const origCallback = videoCountWidget.callback;
                videoCountWidget.callback = function (v) {
                    if (origCallback) origCallback.call(this, v);
                    updateInputs();
                };
            }
            if (audioCountWidget) {
                const origCallback = audioCountWidget.callback;
                audioCountWidget.callback = function (v) {
                    if (origCallback) origCallback.call(this, v);
                    updateInputs();
                };
            }

            // Also poll for value changes (handles arrow buttons and direct input)
            let lastVideoCount = -1;
            let lastAudioCount = -1;
            const checkForChanges = () => {
                const currentVideo = videoCountWidget?.value ?? 0;
                const currentAudio = audioCountWidget?.value ?? 0;
                if (currentVideo !== lastVideoCount || currentAudio !== lastAudioCount) {
                    lastVideoCount = currentVideo;
                    lastAudioCount = currentAudio;
                    updateInputs();
                }
            };

            // Check periodically for changes
            node._ffmpegCheckInterval = setInterval(checkForChanges, 200);

            // Clean up interval when node is removed
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (this._ffmpegCheckInterval) {
                    clearInterval(this._ffmpegCheckInterval);
                }
                if (origOnRemoved) origOnRemoved.call(this);
            };

            // Initial update
            setTimeout(() => updateInputs(), 100);

            // Resize constraint - only apply when no preview
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                if (origOnResize) origOnResize.call(this, size);
                // Don't constrain if there's a preview widget
                if (this._ffmpegVideoWidget) return;
                const minH = this.computeSize()[1];
                if (size[1] > minH + 5) {
                    this.setSize([size[0], minH]);
                }
            };

            setTimeout(() => node.setSize([node.size[0], node.computeSize()[1]]), 100);
        }

        // 2. Single AV Merge node - add preview after execution
        if (node.comfyClass === "FFmpeg_VideoAudioMerge_JX") {
            let expectedHeight = null;

            // Override onExecuted to show preview
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                if (origOnExecuted) origOnExecuted.call(this, output);

                // Support both old format (VIDEO) and new format (videos)
                let filename = null;
                if (output?.VIDEO && output.VIDEO.length > 0) {
                    filename = output.VIDEO[0];
                } else if (output?.videos && output.videos.length > 0) {
                    filename = output.videos[0].filename || output.videos[0];
                }

                if (filename) {
                    // Remove existing preview widget
                    if (node._ffmpegResultPreview) {
                        if (node._ffmpegResultPreview.element) {
                            node._ffmpegResultPreview.element.remove();
                        }
                        const index = node.widgets.indexOf(node._ffmpegResultPreview);
                        if (index > -1) node.widgets.splice(index, 1);
                        node._ffmpegResultPreview = null;
                    }

                    // Create preview container
                    const container = document.createElement("div");
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;margin-top:5px;";

                    const video = document.createElement("video");
                    video.controls = true;
                    video.style.cssText = "width:100%;display:block;";

                    // Try output directory first, then temp directory on error
                    const tryLoadVideo = (type) => {
                        video.src = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=&t=${Date.now()}`;
                    };

                    let triedTemp = false;
                    video.onerror = () => {
                        if (!triedTemp) {
                            triedTemp = true;
                            console.log("[FFmpeg Preview] Output failed, trying temp directory");
                            tryLoadVideo("temp");
                        }
                    };

                    video.onloadedmetadata = () => {
                        const aspectRatio = video.videoHeight / video.videoWidth;
                        const nodeWidth = node.size[0] - 20;
                        const previewHeight = nodeWidth * aspectRatio;
                        expectedHeight = 280 + previewHeight;  // Base height for Single AV Merge widgets
                        node.setSize([node.size[0], expectedHeight]);
                    };

                    // Start with output directory
                    tryLoadVideo("output");

                    container.appendChild(video);
                    node._ffmpegResultPreview = node.addDOMWidget("ffmpeg_result_preview", "div", container, { serialize: false });

                    // Set initial height
                    expectedHeight = 400;
                    node.setSize([node.size[0], expectedHeight]);
                }
            };

            const origOnResize = node.onResize;
            node.onResize = function (size) {
                if (origOnResize) origOnResize.call(this, size);

                // Recalculate height if preview exists
                if (node._ffmpegResultPreview && node._ffmpegResultPreview.element) {
                    const video = node._ffmpegResultPreview.element.querySelector('video');
                    if (video && video.videoHeight && video.videoWidth) {
                        const aspectRatio = video.videoHeight / video.videoWidth;
                        const nodeWidth = size[0] - 20;
                        const previewHeight = nodeWidth * aspectRatio;
                        expectedHeight = 280 + previewHeight;
                        this.setSize([size[0], expectedHeight]);
                        return;
                    }
                }

                // Default constraint
                const minH = expectedHeight || this.computeSize()[1];
                if (size[1] > minH + 5) {
                    this.setSize([size[0], minH]);
                }
            };

            setTimeout(() => node.setSize([node.size[0], node.computeSize()[1]]), 50);
        }

        // 3. URL loading nodes (also support local file paths)
        if (["FFmpeg_LoadVideoFromURL_JX", "FFmpeg_LoadAudioFromURL_JX", "FFmpeg_LoadImageFromURL_JX"].includes(node.comfyClass)) {
            const urlWidget = node.widgets?.find(w => w.name === "url");
            const showPreviewWidget = node.widgets?.find(w => w.name === "show_preview");

            if (urlWidget) {
                urlWidget.computeSize = () => [node.size[0], 45];
                if (urlWidget.inputEl) {
                    urlWidget.inputEl.style.height = "30px";
                    urlWidget.inputEl.style.minHeight = "30px";
                    urlWidget.inputEl.style.maxHeight = "30px";
                    urlWidget.inputEl.style.resize = "none";
                }
            }

            // Track expected height for preview
            let expectedHeight = null;

            const showURLPreview = (url) => {
                if (!url || !url.trim()) {
                    hideURLPreview();
                    return;
                }

                // Remove existing preview - stop playback first to release resources
                if (node._ffmpegUrlPreview) {
                    const oldMedia = node._ffmpegUrlPreview.element?.querySelector('video, audio');
                    if (oldMedia) {
                        oldMedia.pause();
                        oldMedia.src = '';
                        oldMedia.load(); // Force release of old resource
                    }
                    if (node._ffmpegUrlPreview.element) node._ffmpegUrlPreview.element.remove();
                    const index = node.widgets.indexOf(node._ffmpegUrlPreview);
                    if (index > -1) node.widgets.splice(index, 1);
                    node._ffmpegUrlPreview = null;
                }

                const container = document.createElement("div");

                if (node.comfyClass === "FFmpeg_LoadVideoFromURL_JX") {
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;";
                    const video = document.createElement("video");
                    video.controls = true;
                    video.style.cssText = "width:100%;display:block;";
                    // Add timestamp to prevent caching
                    video.src = url.trim() + (url.includes('?') ? '&' : '?') + '_t=' + Date.now();
                    video.onloadedmetadata = () => {
                        const aspectRatio = video.videoHeight / video.videoWidth;
                        const nodeWidth = node.size[0] - 20;  // Account for padding
                        const previewHeight = nodeWidth * aspectRatio;
                        // Base height for other widgets + preview + video controls
                        expectedHeight = 120 + previewHeight;
                        node.setSize([node.size[0], expectedHeight]);
                    };
                    container.appendChild(video);
                    node._ffmpegUrlPreview = node.addDOMWidget("ffmpeg_url_preview", "div", container, { serialize: false });
                } else if (node.comfyClass === "FFmpeg_LoadAudioFromURL_JX") {
                    container.style.cssText = "background:#2a2a2a;border-radius:4px;padding:3px;";
                    const audio = document.createElement("audio");
                    audio.controls = true;
                    audio.style.cssText = "width:100%;height:25px;";
                    audio.src = url.trim();
                    container.appendChild(audio);
                    node._ffmpegUrlPreview = node.addDOMWidget("ffmpeg_url_preview", "div", container, { serialize: false });
                    expectedHeight = node.computeSize()[1];
                    node.setSize([node.size[0], expectedHeight]);
                } else if (node.comfyClass === "FFmpeg_LoadImageFromURL_JX") {
                    container.style.cssText = "width:100%;background:#1a1a1a;border-radius:4px;";
                    const img = document.createElement("img");
                    img.style.cssText = "width:100%;height:auto;display:block;";
                    img.src = url.trim();
                    img.onload = () => {
                        const aspectRatio = img.naturalHeight / img.naturalWidth;
                        const nodeWidth = node.size[0] - 20;  // Account for padding
                        const previewHeight = nodeWidth * aspectRatio;
                        // Base height for other widgets + preview + padding
                        expectedHeight = 140 + previewHeight;
                        node.setSize([node.size[0], expectedHeight]);
                    };
                    container.appendChild(img);
                    node._ffmpegUrlPreview = node.addDOMWidget("ffmpeg_url_preview", "div", container, { serialize: false });
                }
            };

            const hideURLPreview = () => {
                if (node._ffmpegUrlPreview) {
                    if (node._ffmpegUrlPreview.element) node._ffmpegUrlPreview.element.remove();
                    const index = node.widgets.indexOf(node._ffmpegUrlPreview);
                    if (index > -1) node.widgets.splice(index, 1);
                    node._ffmpegUrlPreview = null;
                }
                expectedHeight = null;
                node.setSize([node.size[0], node.computeSize()[1]]);
            };

            const scheduleUpdatePreview = (delay = 300) => {
                clearTimeout(node._ffmpegUrlPreviewTimeout);
                node._ffmpegUrlPreviewTimeout = setTimeout(() => {
                    const shouldShow = showPreviewWidget ? showPreviewWidget.value : true;
                    const url = urlWidget?.value?.trim();
                    if (shouldShow && url) {
                        showURLPreview(url);
                    } else {
                        hideURLPreview();
                    }
                }, delay);
            };

            if (urlWidget) {
                const oCallback = urlWidget.callback;
                urlWidget.callback = function (v) {
                    if (oCallback) oCallback.call(this, v);
                    scheduleUpdatePreview(500);
                };
            }
            if (showPreviewWidget) {
                const opCallback = showPreviewWidget.callback;
                showPreviewWidget.callback = function (v) {
                    if (opCallback) opCallback.call(this, v);
                    scheduleUpdatePreview(0);
                };
            }

            // Resize constraint - prevent dragging to create whitespace
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                if (origOnResize) origOnResize.call(this, size);

                // Recalculate height based on new width when preview exists
                if (node._ffmpegUrlPreview && node._ffmpegUrlPreview.element) {
                    const mediaElement = node._ffmpegUrlPreview.element.querySelector('video, img');
                    if (mediaElement) {
                        let aspectRatio = 0;
                        if (mediaElement.tagName === 'VIDEO') {
                            aspectRatio = mediaElement.videoHeight / mediaElement.videoWidth || 0;
                        } else if (mediaElement.tagName === 'IMG') {
                            aspectRatio = mediaElement.naturalHeight / mediaElement.naturalWidth || 0;
                        }

                        if (aspectRatio > 0) {
                            const nodeWidth = size[0] - 20;
                            const previewHeight = nodeWidth * aspectRatio;
                            const baseHeight = mediaElement.tagName === 'VIDEO' ? 120 : 140;
                            expectedHeight = baseHeight + previewHeight;
                            this.setSize([size[0], expectedHeight]);
                            return;
                        }
                    }
                }

                // Fallback: use expectedHeight or computed size
                const minH = expectedHeight || this.computeSize()[1];
                if (size[1] > minH + 5) {
                    this.setSize([size[0], minH]);
                }
            };

            // Trigger initial preview on node load
            setTimeout(() => {
                scheduleUpdatePreview(100);
            }, 200);
        }
    }
});
