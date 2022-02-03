


[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)   [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://evid.readthedocs.io/en/latest/index.html)


# Evid - pandemic simulator for big cities

Evid is an open-source agent-based epidemic simulator, to simulate spreads of epidemic in big cities.

![Logo](https://raw.githubusercontent.com/bezirganyan/evid/master/logo.png)


## Documentation

[Documentation](https://evid.readthedocs.io/en/latest/)


## Getting started

Clone the project

```bash
  git clone https://github.com/bezirganyan/evid.git
```

Go to the project directory

```bash
  cd evid
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```

Start a somilation run

```bash
  python3 run.py configs/config.yaml 1440
```
where `configs/config.yaml` is the default configuration file,
which can be modified, and the `1440` is the number of simulation steps to perform. Since by default each simulation step corresponds to one real life hour, 1440 steps will generate 1440/24 = 60 days of data. number of simulation steps. By default one simulation step corresponds to one hour in the simulation world.


