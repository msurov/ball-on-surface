#!/bin/bash

ffmpeg -framerate 60 -i frame%04d.jpg -c:v libx265 -crf 20 out.mp4
