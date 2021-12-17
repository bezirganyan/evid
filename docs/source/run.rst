Getting Started
===============

Installation
~~~~~~~~~~~~

You can clone the repository and install the requirements with: ::

    $ git clone https://github.com/bezirganyan/evid.git
    $ pip3 install -r requirements.txt

Running the simulator
~~~~~~~~~~~~~~~~~~~~~

You can run the code with the command::

    python3 run.py configs/config.yaml 1440
where ``configs/config.yaml`` is the default configuration file, which can be modified,
and the ``1440`` is the number of simulation steps to perform. Since by default each simulation
step corresponds to one real life hour, 1440 steps will generate 1440/24 = 60 days of data.
number of simulation steps. By default one simulation step corresponds to one hour in
the simulation world.

.. automodule:: run
   :members:
   :undoc-members:
   :show-inheritance:
