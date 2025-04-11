import cv2
import numpy as np
from collections import OrderedDict

class VideoReader_cv2(object):
    def __init__(self, uri, ctx=None, width=-1, height=-1, num_threads=0, fault_tol=-1):
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"unable to open the file: {uri}")
        self._num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._num_frame <= 0:
            raise RuntimeError(f"invalid video frames: {self._num_frame}")
        self.width = width
        self.height = height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps if self.fps > 0 else 0
        self.frame_cache = OrderedDict()  # 缓存最近读取的帧

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def __len__(self):
        return self._num_frame

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return self.get_batch(indices)
        if idx < 0:
            idx += self._num_frame
        if idx < 0 or idx >= self._num_frame:
            raise IndexError(f"idx {idx} exceed [0, {self._num_frame-1}]")
        self.seek_accurate(idx)
        return self.next()

    def next(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration()
        if self.width > 0 and self.height > 0:
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def seek_accurate(self, pos):
        if pos < 0 or pos >= self._num_frame:
            raise RuntimeError(f"skip position {pos} exceed [0, {self._num_frame-1}]")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def get_batch(self, indices):
        indices = self._validate_indices(indices)
        # 检查缓存
        frames = []
        uncached_indices = []
        for idx in indices:
            if idx in self.frame_cache:
                frames.append(self.frame_cache[idx])
            else:
                uncached_indices.append(idx)
                frames.append(None)  # 占位

        if uncached_indices:
            # 对未缓存的索引排序并顺序读取
            sorted_indices = np.sort(uncached_indices)
            temp_frames = []
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            for idx in sorted_indices:
                if idx < current_pos:
                    self.seek_accurate(idx)
                    current_pos = idx
                else:
                    self.skip_frames(idx - current_pos)
                frame = self.next()
                temp_frames.append(frame)
                self.frame_cache[idx] = frame  # 存入缓存
                current_pos = idx + 1
                # 限制缓存大小
                if len(self.frame_cache) > 100:
                    self.frame_cache.popitem(last=False)

            # 将读取的帧填入结果
            for i, idx in enumerate(indices):
                if frames[i] is None:
                    frames[i] = self.frame_cache[idx]

        return np.stack(frames, axis=0)

    def get_frame_timestamp(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
        else:
            indices = [idx]
        indices = self._validate_indices(indices)
        timestamps = []
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        for i in indices:
            if i < current_pos:
                self.seek_accurate(i)
            else:
                self.skip_frames(i - current_pos)
            start_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 转换为秒
            # 获取下一帧时间戳
            if i + 1 < self._num_frame:
                self.seek_accurate(i + 1)
                end_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                self.seek_accurate(i)  # 恢复位置
            else:
                end_time = self.total_duration
            timestamps.append([start_time, end_time])
            current_pos = i
        return np.array(timestamps)

    def get_avg_fps(self):
        return self.fps

    def skip_frames(self, num=1):
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_pos = current_pos + num
        if new_pos >= self._num_frame:
            new_pos = self._num_frame - 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

    def _validate_indices(self, indices):
        indices = np.array(indices, dtype=np.int64)
        indices[indices < 0] += self._num_frame
        if not (indices >= 0).all() or not (indices < self._num_frame).all():
            raise IndexError("exceed the idx")
        return indices

    def get_key_indices(self):
        return []  # 暂未实现
    
    
class CPUContext:
    """A simple CPU context class to mimic DECORDContext."""
    def __init__(self, device_type, device_id):
        """
        Initialize the CPU context.
        
        Parameters:
        -----------
        device_type : int
            The type of device (1 for CPU).
        device_id : int
            The integer device ID.
        """
        self.device_type = device_type  # 1 indicates CPU
        self.device_id = device_id

    def __str__(self):
        """String representation of the context."""
        return f"cpu({self.device_id})"

def cpu(dev_id=0):
    """
    Construct a CPU device context.

    Parameters:
    -----------
    dev_id : int, optional
        The integer device ID (default is 0).

    Returns:
    --------
    ctx : CPUContext
        The created CPU context object.
    """
    return CPUContext(device_type=1, device_id=dev_id)