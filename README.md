# SORI

This repo contains the GPU and CPU code for SORI, which calls a GPU implementation of DPRF.

Set up the environment on traverse with:

```
source setup_env_traverse.sh
```

Compile with

```
make all
```

Run the resulting sori_main.exe

Allocate a node for testing:
```
salloc -N1 --time=01:00:00 --reservation test --gpus-per-task=1
```
