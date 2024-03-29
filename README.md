# Wasp AL and ML

## Setup host system
```
chmod +x environment/base-packages.sh
sudo ./environment/base-packages.sh
```

## Setup and enter the docker image environment 

In linux

```
./docker.sh [clean]
```

## Setup and enter the virtual environment 

In windows

```
env.bat 
```


In linux

```
source ./env.sh [clean]
```

In docker

```
source ./env.sh [clean]
```

## Update docker environment

In the file (environment/base-packeges.sh) add apt packages that you need in your project

note that a newline will brake the RUN command and thus "\\" should be used when adding dependencies. More information on how docker works can be found on https://docs.docker.com/get-started/


## Update pip environment

Due to some packages accelerator dependencies like Nvidia GPU'S and pytorch
its nice to split packages into three files

*environment\requirements_base.txt* - accelerator independent packages

*environment\requirements_gpu.txt* - accelerator dependent GPU packages

*environment\requirements_base.txt* - accelerator dependent CPU alternatives packages
