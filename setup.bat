@echo off
setlocal

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py -3.13
) else (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo Python 3.13 was not found.
        echo Install Python 3.13 from https://www.python.org/downloads/
        exit /b 1
    )
    set PYTHON=python
)

%PYTHON% -m venv .venv
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

if errorlevel 1 exit /b 1

echo.
echo Setup complete.
echo Run: .venv\Scripts\python.exe app.py
