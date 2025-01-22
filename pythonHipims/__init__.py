import os
__version__ = '1.0.0'

print("         Welcome to the HiPIMS ", __version__)

dir_path = os.path.dirname(os.path.realpath(__file__))

f = open(os.path.join(dir_path, 'banner.txt'), 'r')
file_contents = f.read()
print(file_contents)
f.close()