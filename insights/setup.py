from setuptools import setup, find_packages

setup(
    name="ai-workforce-tracker",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "pandas>=1.3.0",
        "sqlalchemy>=1.4.0",
        "requests>=2.26.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.8",
) 