import os


if os.name == 'nt':
    os.system("python -m oct examples//data//3_BL_Catheter2_human_coronary_artery_ramp_struct_ps tomo+struct+ps+proj mgh 1")
if os.name == 'posix':
    os.system("python -m oct examples//data//3_BL_Catheter2_human_coronary_artery_ramp_struct_ps tomo+struct+ps+proj mgh 1")

