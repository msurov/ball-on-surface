#!/bin/bash

ffmpeg -framerate 60 -i frame%04.png -c:v libx265 out.mp4
