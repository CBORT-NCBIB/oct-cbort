import os

if os.name == 'nt':
    os.system("python -m oct examples//data//1_VL_Benchtop1_rat_nerve_biseg_n2_m5_struct_angio_ps tomo+struct+angio+ps+hsv+proj mgh 1")
    print('================  1/4')
    os.system("python -m oct examples//data//2_BL_Catheter1_rat_clot_ramp_struct_ps tomo+struct+ps+proj mgh 1")
    print('================  2/4')
    os.system("python -m oct examples//data//3_BL_Catheter2_human_coronary_artery_ramp_struct_ps tomo+struct+ps+proj mgh 1")
    print('================  3/4')
    os.system("python -m oct examples//data//4_BL_Benchtop_Phantom_struct_angio_ps tomo+struct+ps+hsv+stokes mgh 1")
    print('================  4/4')
    print('Test Complete')
if os.name == 'posix':
    os.system("python -m oct examples//data//1_VL_Benchtop1_rat_nerve_biseg_n2_m5_struct_angio_ps tomo+struct+angio+ps+hsv+proj mgh 1")
    print('================  1/4')
    os.system("python -m oct examples//data//2_BL_Catheter1_rat_clot_ramp_struct_ps tomo+struct+angio+ps+proj mgh 1")
    print('================  2/4')
    os.system("python -m oct examples//data//3_BL_Catheter2_human_coronary_artery_ramp_struct_ps tomo+struct+ps+proj mgh 1")
    print('================  3/4')
    os.system("python -m oct examples//data//4_BL_Benchtop_Phantom_struct_angio_ps tomo+struct+ps+hsv+stokes mgh 1")
    print('================  4/4')
    print('Test Complete')