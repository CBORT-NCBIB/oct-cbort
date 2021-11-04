import os


if os.name == 'nt':
    os.system("python -m oct examples//data//4_BL_Benchtop_Phantom_struct_angio_ps tomo+struct+ps+hsv+stokes mgh 1")
if os.name == 'posix':
    os.system("python -m oct examples//data//4_BL_Benchtop_Phantom_struct_angio_ps tomo+struct+ps+hsv+stokes mgh 1")