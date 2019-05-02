from itertools import count
from enum import IntEnum
import struct
import json

header_frame = "START"
init = "INIT"
read_header = "READ_HEADER"
read_data = "READ_DATA"

int_len = 4
float_len = 4
float_dtype = '>f4'
int_dtype = '>i4'
header_length = len(header_frame)*2

num_channels = 3
fft_size = 256
width = 48