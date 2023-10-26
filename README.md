# optboolnet
 The optimization toolbox for Boolean network analysis

## Docker distribution

### Build docker image

```
docker build -t mso_lab/colomoto-optboolnet:v0 .
```

### Launch CoLoMoTo Notebook

```
python -m pip install -U colomoto-docker
colomoto-docker --image mso_lab/colomoto-optboolnet -V v0 --bind .
```

Note: `gurobi.lic` must be in the working directoy.
