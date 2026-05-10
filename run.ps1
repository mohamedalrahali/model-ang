# Lance l'application ML (dépendances + artefacts + serveur)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install -r requirements.txt -q
python serve.py @args
