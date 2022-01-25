#!/bin/bash 

# python tools/retrain.py --arch RFSTEP3 --cutout --cutout_length 8 --model_name "retain_rf_resnet_cutout_8"


# python tools/retrain.py --model_base 'resnet20' --model_name 'resnet20_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'resnet32' --model_name 'resnet32_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'resnet44' --model_name 'resnet44_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'resnet56' --model_name 'resnet56_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'resnet110' --model_name 'resnet110_base' --arch 'NORMAL'

# python tools/retrain.py --model_base 'rf_resnet20' --model_name 'rf_resnet20_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet32' --model_name 'rf_resnet20_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet44' --model_name 'rf_resnet44_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet56' --model_name 'rf_resnet56_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'rf_resnet110' --model_name 'rf_resnet110_base_rfstep3' --arch RFSTEP3

# python tools/retrain.py --model_base 'la_resnet20' --model_name 'la_resnet20_base_rfstep3' --arch Attention

# python tools/retrain.py --model_base 'la_resnet32' --model_name 'la_resnet20_base_rfstep3' --arch Attention

# python tools/retrain.py --model_base 'la_resnet44' --model_name 'la_resnet44_base_rfstep3' --arch Attention

python tools/retrain.py --model_base 'la_resnet56' --model_name 'la_resnet56_base_rfstep3' --arch Attention

# python tools/retrain.py --model_base 'la_resnet110' --model_name 'la_resnet110_base_rfstep3' --arch Attention





# module load  cuda/10.1
# module load cudnn/7.6.5.32_cuda10.1u2

# module load anaconda
# source activate torch17 

# 1.20日 18:30开始运行程序