
try:
    import cupy
    print("CuPy imported successfully")
except ImportError as e:
    print(f"CuPy import failed: {e}")

try:
    from numba import cuda
    print(f"Numba CUDA available: {cuda.is_available()}")
except ImportError as e:
    print(f"Numba import failed: {e}")
except Exception as e:
    print(f"Numba CUDA check failed: {e}")
