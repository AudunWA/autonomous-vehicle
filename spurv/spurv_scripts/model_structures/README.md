# Why these exists
The SPURV uses ROS, which only runs on Python 2.
We train and serialize our models in a Python 3 environment, which causes incompatibility issues when using `Lambda`-layers.
To make these models work, we have to define them identically in our Python 2 code, and then load the weights only.