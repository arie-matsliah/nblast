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
```

# Get data
Place swc files under `./swc/`

# Run
```
python -m run_nblast
```