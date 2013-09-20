Introduction
============
Thank you for downloading SpikeGPU software. You can find out more information
about SpikeGPU by visiting our website: http://spikegpu.sbel.org

Compilation
===========
For the example program, we have supplied an example CMakeList file for you
(We will provide method for compilation on Visual Studio soon). To compile
using the CMakeList file, the following steps can be followed:
(1) Create an empty folder in an arbitrary position;
(2) Do ``ccmake $(SPIKE_LIB_PATH)/examples'';
(3) Specify the flags, do the configuration and generate the Makefile;
(4) Do ``make'' to generate the executable.

Usage
=====
For detailed usage of the example program, just do
``$(EXECUTABLE_PATH)/$(EXECUTABLE_NAME) [-h|-?|--help]''

Files
=====
README         -  This file
LICENCE        -  Licence agreement
spike/         -  Including all header files to use the software
examples/      -  Including all example programs (currently only support solving systems in matrix market format)
