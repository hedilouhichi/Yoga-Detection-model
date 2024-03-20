from cx_Freeze import setup, Executable

setup(
    name="YOGA POSTURE DETECTOR",
    version="1.0",
    description="YOGA POSTURE DETECTOR",
    options={"build_exe": {"build_dir": "C:/Users/user/Desktop"}},
    executables=[Executable("yoga.py")],
)
