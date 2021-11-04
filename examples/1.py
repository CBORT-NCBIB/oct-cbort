import os


if os.name == 'nt':
    os.system("python -m oct examples//data//1_VL_Benchtop1_rat_nerve_biseg_n2_m5_struct_angio_ps tomo+struct+angio+ps+hsv+proj mgh 1")
elif os.name == 'posix':
    os.system("python -m oct examples//data//1_VL_Benchtop1_rat_nerve_biseg_n2_m5_struct_angio_ps tomo+struct+angio+ps+hsv+proj mgh 1")
