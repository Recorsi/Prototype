

- Command to run the docker:
``` docker run -it -p 5000:5000 --rm -v  /home/hpoojari/Documents/deepfake/Prototype/Samplevideo/:/app/uploads -v /home/hpoojari/Documents/deepfake/Prototype/output:/app/outputs deepfake-detector /bin/bash ```
- After having access to the container shell you can access your application in 2 way using UI or running the script.
1. For UI access run this command inside the container shell: ``` python3 main.py```
2. For directly running the script in the container use thsi cmd: ```python3 detect_from_video.py -i uploads -o output```

if you want to build the image, clone the repo and the run ``` docker build -t deepfake-detector .```
