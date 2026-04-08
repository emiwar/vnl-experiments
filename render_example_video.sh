python vnl_experiments/tools/camera_matched_rollout.py \
  --checkpoint downloaded_checkpoints/Imitation_detached_critic_v4-20260407-161629/ \
  --calibration assets/art/2020_12_22_1/calibration/hires_cam4_params.mat \
  --video assets/art/2020_12_22_1/videos/Camera4/0.mp4 \
  --reference_h5 assets/art/2020_12_22_1/transform_art_2020_12_22_1.h5 \
  --video_offset 0 \
  --start_time 15 --end_time 255 \
  --n_rollouts 1 \
  --output rollouts/camera4_15_to_255_with_high_force_cost.mp4
