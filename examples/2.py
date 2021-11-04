import os

if os.name == 'nt':
    os.system("python -m oct examples//data//2_BL_Catheter1_rat_clot_ramp_struct_ps tomo+struct+ps+proj mgh 1")
elif os.name == 'posix':
    os.system("python -m oct examples//data//2_BL_Catheter1_rat_clot_ramp_struct_ps tomo+struct+angio+ps+proj mgh 1")


