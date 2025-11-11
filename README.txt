to create the environment run:
    python -m venv env

to activate the environment in git bash run:
    env\Scripts\activate         

to inisially install all the requirementsin the env run:
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

to activate the environment in git bash run:
    deactivate

to run ansys sim run:
    cd sim
    python run_two_cubes_from_inp.py --inp cubes.inp   --anchor cube=1,face=-X   --force cube=2,face=+X,type=nodal,value=1e6   --solve

to kill any ansys Process
    Get-Process | Where-Object { $_.ProcessName -like "ansys*" -or $_.ProcessName -like "MAPDL*" -or $_.ProcessName -like "fluent*" -or $_.ProcessName -like "launcher*" } | Stop-Process -Force


conclustions:
    the base and length or the arm should be fixed dims