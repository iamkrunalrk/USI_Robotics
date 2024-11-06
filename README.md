# Waste Detection and Classification Robot

## Usage
```sh
pip install -r requirements.txt
```

In CoppeliaSim load the scene

```sh
ros2 launch thymioid main.launch device:="tcp:host=localhost;port=33333" simulation:=True name:=thymio0
```

And finally launch the node

```sh
cd code/scripts
python3 thymio_random.py
```
