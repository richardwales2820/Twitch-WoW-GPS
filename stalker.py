import streamlink
import cv2
import subprocess as sp
import numpy as np

class Stream:
    def __init__(self, url):
        self.stream = streamlink.streams(url)['720p']
        self.pipe = sp.Popen(['/usr/local/bin/ffmpeg', "-i", 
            self.stream.url,
            "-loglevel", "quiet",  # no text output
            "-an",  # disable audio
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "-"],
            stdin=sp.PIPE, stdout=sp.PIPE)
    
    def read_default_map_location(self):
        raw_image = self.pipe.stdout.read(1280 * 720 * 3)
        frame = np.fromstring(raw_image, dtype='uint8').reshape((720, 1280, 3))
        cropped_location = frame[0:20:, -150:-10]

        return cv2.cvtColor(cropped_location, cv2.COLOR_BGR2RGB)

    def read_frame(self):
        raw_image = self.pipe.stdout.read(1280 * 720 * 3)
        frame = np.fromstring(raw_image, dtype='uint8').reshape((720, 1280, 3))

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

stream = Stream('https://www.twitch.tv/sodapoppin')

frame = stream.read_frame()
map_frame = stream.read_default_map_location()

cv2.imshow('Whole_Image', frame)
cv2.imshow('Map_Image', map_frame)

cv2.waitKey(0)