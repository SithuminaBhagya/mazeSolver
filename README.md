# mazeSolver
In this project, a method is presented to automatically parse a 2D maze image and generate a path for a robot traveling from the maze entrance at the top to the exit at the bottom. Python program language has been used to develop the algorithm. 

First Import the necessary libraries
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation

Run sampleSolution.py for the whole program. This will animate the BFS and solve the maze.
To view the pixel map in binary of the maze run basicImage.py -----> This gives the not-cropped version of the pixel map in Binary
In the sampleSolution.py
        uncommment lines 41-42 to see the cropped pixel matrix in binary.
        uncommment lines 1748-180 to see the connection of the cells. This will print out the open_closed_paths dictionary.
        change the animation time by changing the interval on line 306
