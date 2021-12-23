python test.py --root_path /e/data --video_path UCF101/jpg --annotation_path UCF101/ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path /d/Files/GitHub/3D-ResNets-Paddle/checkpoint/resnet-18-kinetics.pth \
--model resnet --model_depth 18 --resnet_shortcut A --batch_size 64 --n_threads 4 --checkpoint 5