### CarND-Behavioral-Cloning-P3

#### Purpose
This project aims to demonastrate that with a suitably trained model, one can transfer the knowledge gained from driving around a specific track (using a simulator) can then be used by a non-human driver to drive the car autonomously in a very different track.

#### Prerequsites
Track Traning Data provided by Udacity  
Keras 1.2.0 or above  

#### Code organisation
The code assumes that there is a directory named 'data' where the python notebook is being run.  
This is the Track Traning Data provided by Udacity.  
While interactive sessions are conducted via an ipynb, it is also then downloaded as model.py.  
File model.json is the json representation of the model chosen and model.h5 represents the weights saved to disk.  
File drive.py uses socketio to send angles/throttle to the car in the simuator, and uses the model.json and model.h5 files.  
