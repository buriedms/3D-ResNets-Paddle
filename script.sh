#使用 avi 转换为 jpg 文件 utils/video_jpg_ucf101_hmdb51.py
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory

#使用生成 n_frames 文件 utils/n_frames_ucf101_hmdb51.py  ./data/UCF101
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory

#使用类似于 ActivityNet 的 json 格式生成注释文件 ./data/UCF101
python utils/ucf101_json.py annotation_dir_path

#确认所有选项。pyt
python main.lua -h

#在具有 4 个 CPU 线程（用于数据加载）的 Kinetics 数据集（400 个类）上训练 ResNets-34。
#批量大小为 128。
#每 5 个 epoch 保存一次模型。
python main.py --root_path ./data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5

#从 epoch 101 继续训练。（加载了 ~/data/results/save_100.pth。）
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5

## 在 UCF-101 上微调预训练模型 (~/data/models/resnet-34-kinetics.pth) 的 conv5_x 和 fc 层。
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5

