@echo off

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Create necessary directories
echo Creating project directories...
mkdir data_cache
mkdir logs

echo Creating data directory...
mkdir data

echo Setup complete! Virtual environment is activated.
echo To deactivate the virtual environment, run: deactivate
echo Run 'python -m uvicorn app:app --reload' to start the application 