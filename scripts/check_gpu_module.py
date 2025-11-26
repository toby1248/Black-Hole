
try:
    import tde_sph.gpu
    print(f"tde_sph.gpu imported. HAS_CUDA={tde_sph.gpu.HAS_CUDA}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
