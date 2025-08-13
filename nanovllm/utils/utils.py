import gc

# A context manager to control the state of the garbage collector.
# When entering the context, it sets the GC state to the specified state (enabled or disabled).
# When exiting the context, it restores the GC state to its previous state.
# also can be use as a decorator to wrap functions to control GC state during their execution.
class gc_control:
    def __init__(self, enable: bool):
        self.enable = enable
        self.prev_state = None

    def __enter__(self):
        self.prev_state = gc.isenabled()
        if self.enable:
            gc.enable()
        else:
            gc.disable()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev_state is not None:
            if self.prev_state:
                gc.enable()
            else:
                gc.disable()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


# disable_gc: A decorator or context manager to disable garbage collection during the execution of a function or a block of code.
disable_gc = lambda enable=True: gc_control(enable=not enable)

# enable_gc: A decorator or context manager to enable garbage collection during the execution of a function or a block of code.
enable_gc = lambda enable=True: gc_control(enable=enable)
