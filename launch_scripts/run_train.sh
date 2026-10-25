#1- Parse images into mp4 videos its at 6FPS (important to ensure aligned groundtruth)
DATA_DIR=''
OUT_DIR=''

python3 scripts/parse_imgs_to_mp4.py --in_dir ${DATA_DIR}/MeVIS/ --split "train" --out_dir ${OUT_DIR}/MeVIS
python3 scripts/parse_imgs_to_mp4.py --in_dir ${DATA_DIR}/MeVIS/ --split "valid_u" --out_dir ${OUT_DIR}/MeVIS
python3 scripts/parse_imgs_to_mp4.py --in_dir ${DATA_DIR}/ReVOS/ --split "train" --out_dir ${OUT_DIR}/ReVOS
python3 scripts/parse_imgs_to_mp4.py --in_dir ${DATA_DIR}/RVOS/ --split "train" --out_dir ${OUT_DIR}/Ref-YT-VOS/

#### That generation of json masks rle should have happened in Download function of Molmo2 code but I separated it outside for multiple reasons
python3 scripts/parse_masks_to_rle.py --in_dir ${DATA_DIR} --split "valid_u" --out_dir ${OUT_DIR}

#2- Visualize Datasets
python3 scripts/dataset_visualize.py mevis_ground vis_html/ --num_examples 50
python3 scripts/dataset_visualize.py mevis_track vis_html/ --num_examples 50
python3 scripts/dataset_visualize.py mocentric_mevis_ground vis_html/ --num_examples 50
python3 scripts/dataset_visualize.py revos_ground vis_html/ --num_examples 50
python3 scripts/dataset_visualize.py revos_track vis_html/ --num_examples 50
python3 scripts/dataset_visualize.py ref_yt_vos_track vis_html/ --num_examples 50

# 3- LoRA Tuning
# wandb offline (for offline logging), wandb sync PATH (syncing offline run), wandb leet PATH (console monitoring for offline runs)
export HF_HOME=''
export MOLMO_DATA_DIR=''
CKPT_DIR=''
HF_OUT_DIR=''

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
export MASTER_PORT='12235'
export PYTHONPATH=""
# Run experiment of Molmo2 baseline, just finetuning for more iterations without motion-centric data
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/sft.py 'default_ckpt/Molmo2-8B/' 'refvos' --save_folder="${CKPT_DIR}/new_save_more_mevis/" --lora='vit' --device_batch_size=1 --num_workers=0 --model.mm_preprocessor.video.max_frames=48 --model.mm_preprocessor.image.max_images=1 --seq_len=12000 --model.llm.max_sequence_length=12000 --cp_degree=1 --inf_eval_interval=500 --n_iterations=3000

# Run experiment of Molmo2*
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python launch_scripts/sft.py 'default_ckpt/Molmo2-8B/' 'refvos_mocentric' --save_folder="${CKPT_DIR}/new_save_mocentric_mevis/" --lora='vit' --device_batch_size=1 --num_workers=0 --model.mm_preprocessor.video.max_frames=48 --model.mm_preprocessor.image.max_images=1 --seq_len=12000 --model.llm.max_sequence_length=12000 --cp_degree=1 --inf_eval_interval=500 --n_iterations=3000

# Convert hugging face ckpt
python -m olmo.hf_model.convert_molmo2_to_hf ${CKPT_DIR}/new_save_mocentric_mevis/step3000/ ${HF_OUT_DIR} --use_bfloat16 --lora 'vit'
