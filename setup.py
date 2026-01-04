from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audio_avif",
    version="0.1.0",
    author="User",
    description="Experimental audio compression using AVIF image encoding of Mel spectrograms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    py_modules=["compress_audio"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "soundfile",
        "pillow",
        "pillow-avif-plugin",
    ],
    entry_points={
        "console_scripts": [
            "audio-avif=compress_audio:main",
        ],
    },
)
