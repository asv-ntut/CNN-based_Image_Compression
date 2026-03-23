import os
import subprocess

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent

package_name = "compressai"
version = "1.2.0"
git_hash = "unknown"


try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    pass


def write_version_file():
    path = cwd / "src" / package_name / "version.py"
    with path.open("w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'git_version = "{git_hash}"\n')


write_version_file()


def get_extensions():
    ext_dirs = cwd / "src" / package_name / "cpp_exts"
    ext_modules = []

    # Add rANS module
    rans_lib_dir = cwd / "src" / "ryg_rans"
    rans_ext_dir = ext_dirs / "rans"

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        # 🚀 [修改點] 針對 PetaLinux (ARM Cortex-A53) 的編譯器最佳化參數
        extra_compile_args += [
            "-O3",                   # 最高層級的執行速度最佳化
            "-mcpu=cortex-a53",      # 指定目標 CPU 架構，開啟對應的硬體指令
            "-mtune=cortex-a53",     # 針對 Cortex-A53 管線進行指令排程最佳化
            "-ftree-vectorize",      # 強制將迴圈展開並轉換為 NEON 向量指令 (SIMD)
            "-ffast-math"            # 放寬 IEEE 浮點數標準，換取極致的運算速度
        ]

    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules


TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "flake8-comprehensions",
    "isort",
    "mypy",
]


def get_extra_requirements():
    extras_require = {
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "furo"],
        "tutorials": ["jupyter", "ipywidgets"],
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


setup(
    name=package_name,
    version=version,
    description="A PyTorch library and evaluation platform for end-to-end compression research",
    url="https://github.com/InterDigitalInc/CompressAI",
    author="InterDigital AI Lab",
    author_email="compressai@interdigital.com",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("tests",)),
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch>=1.7.1",
        "torchvision",
        "pytorch-msssim",
    ],
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
)