--------------------
# LICENSE FOR USE 

model21cm
Copyright (C) 2019  I. Davenport, N. DePorzio

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/gpl/>.
 
--------------------

# PHYS 201: Data Analysis for Physicists
## Final Project: model21cm 

Authors: I. Davenport, N. DePorzio
Package Name: model21cm


This package infers the parameters for a model consisting of three parts: 

* The emission of 21cm light emitted from an excited state of interstallar hydrogen that has propagated according to cosmological models to an observer on Earth
* The affect of astrophysical foregrounds on the 21cm signal described above
* A noise term additional to the two affects above

--------------------
# USER GUIDE
--------------------

## Package Installation

This packaged, `model21cm`, is published to PyPi and can be installed via `pip` and then imported for use in a Python environment like so: 

```
$ pip install model21cm
>>> import model21cm
```

## Data File Format

Data files should be specified as a comma-separated (.csv) file. The data file should contain up to three columns of data: 

* Column 1: Frequency in [MHz]
* Column 2: Sky Brightess Temperature in [K]
* Column 3: This column is optional. If specified, it should specify whether the data point on the corresponding row is good (value 1) or bad/non-physical (value 0) 

Again, the third data column is optional. If it isn't specified, all data points in the file will be considered "good" and utilized in the analysis. 

The first row of the data file should specify a string label for each data column.

## Importing a Data File

Datasets are handled by the `model21cm.data` class. To import a dataset for analysis, there are two options: 

1) Import with instantiation of `model21cm.data` class. Pass the path to the datafile (or just the datafile name if that file is included with the distribution) as argument to class instantiation: 

```
>>> import model21cm as m21
>>> dataset_1 = m21.data("edges_data.csv")
>>> dataset_2 = m21.data("path/to/dataset.csv")
```

2) Add to an instantiation of the `model21cm.data` class: 

```
>>> import model21cm as m21
>>> dataset_1 = m21.data()
>>> dataset_2 = m21.data()
>>> dataset_1.load_package_data("edges_data.csv") 
>>> dataset_2.load_user_data("path/to/data.csv")
```
