#### Building the Docker image:

The image can be built with the command below in the project directory:
    
     docker build -t featuremethods ./

#### To run the Docker image:

The image can be run by executing the command below:

     docker run -ti -v ${PWD}/figures:/app/figures featuremethods
