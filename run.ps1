# run.ps1 - run src.main with ./src in PYTHONPATH

# Add ./src to PYTHONPATH
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
# Add ./scripts to PYTHONPATH
$env:PYTHONPATH = "$PWD\scripts;$env:PYTHONPATH"

# Run Python module with all CLI arguments
set HYDRA_FULL_ERROR=1
python -m main $args
