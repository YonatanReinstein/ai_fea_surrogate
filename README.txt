to activate the environment in git bash run:
    env\Scripts\activate         
to activate the environment in git bash run:
    deactivate
to kill any ansys Process
    Get-Process | Where-Object { $_.ProcessName -like "ansys*" -or $_.ProcessName -like "MAPDL*" -or $_.ProcessName -like "fluent*" -or $_.ProcessName -like "launcher*" } | Stop-Process -Force
conclustions:
    the base and length or the arm should be fixed dims


setup for windows 11:
install python 3.10 at if not allredy installed:
    https://www.python.org/downloads/release/python-3100/

clone repo by runing:
    git clone https://github.com/YonatanReinstein/ai_fea_surrogate

got to the repo by running:
     cd ai_fea_surrogate

create a vertual anvironment by running:
    py -3.10 -m venv env

enter the vertual environment by running:
    env\Scripts\activate 

install all requirements in the env by running:
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html



