import os


# --- Helper File Wrapper ---
class FileObjectWrapper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self._file = None

    def open(self, mode="rb"):
        self._file = open(self.file_path, mode)
        return self

    def read(self, *args, **kwargs):
        return self._file.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self._file.tell(*args, **kwargs)

    def close(self):
        if self._file:
            self._file.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
