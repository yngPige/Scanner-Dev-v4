from setuptools import setup, find_packages

setup(
    name="blacks-trading-terminal",
    version="1.0.0",
    description="Professional-grade, modular cryptocurrency analytics and trading research platform.",
    author="3lackhands",
    author_email="your.email@example.com",
    url="https://github.com/yngPige/Scanner-Dev-v4",
    packages=find_packages(where=".", exclude=["tests*", "backtest_results*", "Data/historical*", ".venv*"]),
    include_package_data=True,
    install_requires=["0x00000@gmail.com
        # You can paste the requirements here or read from requirements.txt
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "blacks-terminal=main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 