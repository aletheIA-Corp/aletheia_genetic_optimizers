from setuptools import setup, find_packages

setup(
    name="aletheia_genetic_optimizers",
    version="0.2.0",
    author="Aletheia_corp",
    author_email="dsarabiatorres@gmail.com",
    description="Librería para soluciones de algoritmos genéticos de optimización",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aletheIA-Corp/aletheia_genetic_optimizers",
    packages=find_packages(),  # Busca automáticamente todos los paquetes
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "constants_and_tools==0.1.0",
        "plotly==6.0.1"
    ],
    python_requires=">=3.10",
)
