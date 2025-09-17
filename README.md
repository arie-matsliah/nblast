# Setup
```
sudo apt-get update
sudo apt-get install -y r-base
R
```

## Inside R shell
```
install.packages("nat")
install.packages("nat.nblast")
install.packages("reticulate")
q()
```

## Inside python env (e.g. conda)
```
pip install rpy2
pip install navis
pip install plotly
```

# Get data
Place swc files under `./swc/`

# Run
```
python -m run_nblast swc/file1.swc swc/file2.swc 
```