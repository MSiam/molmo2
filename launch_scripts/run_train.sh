#1- Parse images into mp4 videos its at 6FPS (important to ensure aligned groundtruth)
#python3 scripts/parse_imgs_to_mp4.py --in_dir /media/ivulab/alkhwarizmi1/vid_referring_expressions/MeVIS/MeViS_release/ --split "train" --out_dir /media/ivulab/riemann2/molmo2_data/video_datasets/video_track/MeViS/
#python3 scripts/parse_imgs_to_mp4.py --in_dir /media/ivulab/alkhwarizmi1/vid_referring_expressions/MeVIS/MeViS_release/ --split "valid_u" --out_dir /media/ivulab/riemann2/molmo2_data/video_datasets/video_track/MeViS/
#### That generation of json masks rle should have happened in Download function of Molmo2 code but I separated it outside for multiple reasons
#python3 scripts/parse_masks_to_rle.py --in_dir /media/ivulab/alkhwarizmi1/vid_referring_expressions/MeVIS/MeViS_release/ --split "valid_u" --out_dir /media/ivulab/riemann2/molmo2_data/video_datasets/video_track/MeViS/

#python3 scripts/parse_imgs_to_mp4.py --in_dir /media/ivulab/alkhwarizmi1/vid_referring_expressions/ReVOS/ --split "train" --out_dir /media/ivulab/riemann2/molmo2_data/video_datasets/video_track/ReVOS/
#python3 scripts/parse_imgs_to_mp4.py --in_dir /media/ivulab/alkhwarizmi1/vid_referring_expressions/RVOS/ --split "train" --out_dir /media/ivulab/riemann2/molmo2_data/video_datasets/video_track/Ref-YT-VOS/

#2- Visualize Datasets
#python3 scripts/dataset_visualize.py mevis_ground vis_html/ --num_examples 50
#python3 scripts/dataset_visualize.py mevis_track vis_html/ --num_examples 50
#python3 scripts/dataset_visualize.py mocentric_mevis_ground vis_html/ --num_examples 50
#python3 scripts/dataset_visualize.py revos_ground vis_html/ --num_examples 50
#python3 scripts/dataset_visualize.py revos_track vis_html/ --num_examples 50
#python3 scripts/dataset_visualize.py ref_yt_vos_track vis_html/ --num_examples 50

# 3- LoRA Tuning
# wandb offline (for offline logging), wandb sync PATH (syncing offline run), wandb leet PATH (console monitoring for offline runs)
export HF_HOME=/media/ivulab/riemann2/huggingface_cache_temp/
export MOLMO_DATA_DIR=/media/ivulab/riemann2/molmo2_data/
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
export MASTER_PORT='12235'
export PYTHONPATH=""
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/sft.py 'default_ckpt/Molmo2-8B/' 'refvos' --save_folder="/media/ivulab/riemann2/molmo2_ckpt/new_save_more_mevis/" --lora='vit' --device_batch_size=1 --num_workers=0 --model.mm_preprocessor.video.max_frames=48 --model.mm_preprocessor.image.max_images=1 --seq_len=12000 --model.llm.max_sequence_length=12000 --cp_degree=1 --inf_eval_interval=500 --n_iterations=3000 #--debug_no_load_weights #--max_inf_eval_examples=10
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/sft.py 'default_ckpt/Molmo2-8B/' 'refvos_mocentric' --save_folder="/media/ivulab/riemann2/molmo2_ckpt/new_save_mocentric_mevis/" --lora='vit' --device_batch_size=1 --num_workers=0 --model.mm_preprocessor.video.max_frames=48 --model.mm_preprocessor.image.max_images=1 --seq_len=12000 --model.llm.max_sequence_length=12000 --cp_degree=1 --inf_eval_interval=500 --n_iterations=3000 #--debug_no_load_weights #--max_inf_eval_examples=10
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/sft.py 'default_ckpt/Molmo2-8B/' 'refvos' --save_folder="/media/ivulab/riemann2/molmo2_ckpt/new_save_more_mevis/" --lora='vit' --device_batch_size=1 --num_workers=0 --model.mm_preprocessor.image.max_images=1 --cp_degree=1 --inf_eval_interval=2 --n_iterations=100 #--debug_no_load_weights #--max_inf_eval_examples=10
#
#python -m olmo.hf_model.convert_molmo2_to_hf default_ckpt/Molmo2-8B/step2000/ /media/ivulab/riemann2/huggingface_cache_temp/MyMolmo2_Original/ --use_bfloat16
python -m olmo.hf_model.convert_molmo2_to_hf default_ckpt/new_save_mocentric_mevis/step3000/ /media/ivulab/riemann2/huggingface_cache_temp/MyMolmo2_MicentricMeVIS/ --use_bfloat16 --lora 'vit'

# 4- Evaluation
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/eval.py 'default_ckpt/Molmo2-8B/' --task="mevis_track_eval_1fps" --save_to_checkpoint_dir --num_workers 0 #--debug_no_load_weights #--max_examples=10
