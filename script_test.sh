
# forward prediction
CUDA_VISIBLE_DEVICES=0 python codec.py eval /home/yangjy/dataset/HEVC_B_png/ --ckpt_dir model_psnr.pth.tar -o /home/yangjy/dataset/out/HEVC_B/tmm_psnr_fwd_q0_g32_f96/ --cuda --gop_size 32 -f 96 --factor 0 

# bi-directional prediction
CUDA_VISIBLE_DEVICES=0 python codec.py eval /home/yangjy/dataset/HEVC_B_png/ --ckpt_dir model_psnr.pth.tar -o /home/yangjy/dataset/out/HEVC_B/tmm_psnr_bid_q0_g32_f96/ --cuda --gop_size 32 -f 96 --factor 0 --bid

# adaptive prediction direction decision
CUDA_VISIBLE_DEVICES=0 python codec.py eval /home/yangjy/dataset/HEVC_B_png/ --ckpt_dir model_psnr.pth.tar -o /home/yangjy/dataset/out/HEVC_B/tmm_psnr_adp_q0_g32_f96/ --cuda --gop_size 32 -f 96 --factor 0 --adp
