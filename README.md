Command to run the docker:
``` docker run -it -p 5000:5000 --rm -v  /home/hpoojari/Documents/deepfake/Prototype/Samplevideo/:/app/uploads -v /home/hpoojari/Documents/deepfake/Prototype/output:/app/outputs deepfake-detector /bin/bash ```
After having access to the container shell you can access your application in 2 way using UI or running the script.
For UI access run this command inside the container shell: ``` python3 main.py```
For directly running the script in the container use thsi cmd: ```python3 detect_from_video.py -i uploads -o output```
