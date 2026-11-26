try:
    from .blend import blend_images_cy
except ImportError:
    # Extension not pre-built (see setup_blend.py): compile on first use. Needs Cython and a C compiler.
    import pyximport

    pyximport.install()
    from .blend import blend_images_cy