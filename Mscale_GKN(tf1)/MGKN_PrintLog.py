"""
@author: LXA
 Date: 2021 年 6 月 5 日
"""
import MGKN_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    MGKN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    MGKN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)

    MGKN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)
    MGKN_tools.log_string('Loss function: L2 loss\n', log_fileout)
    MGKN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    MGKN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        MGKN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        MGKN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    MGKN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    MGKN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['activate_stop'] != 0:
        MGKN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        MGKN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)
