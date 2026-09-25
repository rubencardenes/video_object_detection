"""Optional TensorRT runtime with reusable CUDA buffers (one serialized caller)."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def engine_payload(data: bytes) -> bytes:
    """Accept raw plans and Ultralytics plans with a length-prefixed JSON header."""
    size = int.from_bytes(data[:4], "little")
    if 0 < size < len(data) - 4:
        try:
            metadata = json.loads(data[4 : 4 + size])
        except (ValueError, UnicodeDecodeError):
            pass
        else:
            if isinstance(metadata, dict):
                return data[4 + size :]
    return data


class TensorRTSession:
    def __init__(self, path: Path):
        try:
            import tensorrt as trt
            from cuda.bindings import runtime as cuda
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT support requires: uv sync --extra tensorrt"
            ) from exc
        self._cuda = cuda
        self._buffers = {}
        self._stream = None
        self.last_inference_ms = None
        self._check(cuda.cudaSetDevice(0))
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        trt.init_libnvinfer_plugins(self._logger, "")
        self._engine = self._runtime.deserialize_cuda_engine(engine_payload(path.read_bytes()))
        if self._engine is None:
            raise RuntimeError(
                f"Cannot load TensorRT engine {path.name}. Check that the engine was built "
                "for this GPU and the installed TensorRT version."
            )
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        inputs = [n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != 1:
            raise ValueError("TensorRT detector must have exactly one image input")
        name = inputs[0]
        shape = tuple(self._engine.get_tensor_shape(name))
        if -1 in shape:
            shape = tuple(self._engine.get_tensor_profile_shape(name, 0)[1])
            if not self._context.set_input_shape(name, shape):
                raise ValueError(f"Cannot set TensorRT input shape to {shape}")
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"Expected batch-one NCHW RGB input, got {shape}")
        self._inputs, self._outputs = [], []
        for name in names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(d <= 0 for d in shape):
                raise ValueError(f"Unsupported data-dependent TensorRT output: {name} {shape}")
            if self._engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Expected device tensor: {name}")
            if self._engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Expected linear TensorRT tensor: {name}")
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            if dtype not in (np.dtype("float32"), np.dtype("float16")):
                raise ValueError(f"Unsupported TensorRT tensor dtype: {name} {dtype}")
            host = np.empty(shape, dtype=dtype)
            pointer = self._check(cuda.cudaMalloc(host.nbytes))
            self._buffers[name] = (host, pointer)
            if not self._context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Cannot bind TensorRT tensor {name}")
            meta = SimpleNamespace(name=name, shape=list(shape), type='tensor(float)' if dtype == np.float32 else 'tensor(float16)')
            (self._inputs if name in inputs else self._outputs).append(meta)
        self._stream = self._check(cuda.cudaStreamCreate())

    @staticmethod
    def _check(result):
        status, *values = result
        if int(status) != 0:
            raise RuntimeError(f"CUDA operation failed: {status}")
        return values[0] if values else None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["TensorRT"]

    def run(self, output_names, feed):
        cuda = self._cuda
        self._check(cuda.cudaSetDevice(0))  # Qt workers may run on different threads.
        name = self._inputs[0].name
        host, pointer = self._buffers[name]
        tensor = np.ascontiguousarray(feed[name], dtype=host.dtype)
        if tensor.shape != host.shape:
            raise ValueError(f"TensorRT expects {host.shape}, got {tensor.shape}")
        self.last_inference_ms = None
        self._check(cuda.cudaMemcpy(pointer, tensor.ctypes.data, tensor.nbytes, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
        started = perf_counter()
        if not self._context.execute_async_v3(stream_handle=self._stream):
            raise RuntimeError("TensorRT inference failed")
        self._check(cuda.cudaStreamSynchronize(self._stream))
        self.last_inference_ms = (perf_counter() - started) * 1000
        outputs = []
        for name in output_names or [out.name for out in self._outputs]:
            host, pointer = self._buffers[name]
            self._check(cuda.cudaMemcpy(host.ctypes.data, pointer, host.nbytes, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            outputs.append(host.copy())
        return outputs

    def close(self):
        cuda = getattr(self, "_cuda", None)
        if cuda is None:
            return
        cuda.cudaSetDevice(0)
        if self._stream is not None:
            cuda.cudaStreamSynchronize(self._stream)
        self._context = None
        for _, pointer in self._buffers.values():
            cuda.cudaFree(pointer)
        self._buffers.clear()
        if self._stream is not None:
            cuda.cudaStreamDestroy(self._stream)
            self._stream = None
        self._engine = None
        self._runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter shutdown or a partially constructed session.
