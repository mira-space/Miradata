import os
import pandas as pd
import torch
import clip
import subprocess
import numpy as np
import imageio.v3 as iio
from easydict import EasyDict as edict
from evaluation.consistency_3D import EvaluateErrBetweenTwoImage
from evaluation.temporal_dino_consistency import EvaluateTemporalDinoConsistency
from evaluation.temporal_clip_consistency import EvaluateTemporalClipConsistency
from evaluation.motion_smoothness import EvaluateMotionSmoothness,MotionSmoothness
from evaluation.dynamic_degree import EvaluateDynamicDegree,DynamicDegree
from evaluation.aesthetic_quality import get_aesthetic_model,EvaluateLaionAesthetic
from evaluation.imaging_quality import MUSIQ,EvaluateImagingQuality
from evaluation.text_video_consistency import ViCLIP, SimpleTokenizer, EvaluateTextVideoConsistency

class metrics_calculator():
    def __init__(self,metrics,ckpt_path="data/ckpt",device="cuda"):
        print(f"Initializing metrics: {metrics}")
        self.ckpt_path=ckpt_path
        self.device=device
        if "temporal_dino_consistency" in metrics:
            self.temporal_dino_consistency_dino_model = torch.hub.load(repo_or_dir='facebookresearch/dino:main',source='github', model='dino_vitb16').to(self.device)
        if "temporal_clip_consistency" in metrics:
            self.temporal_clip_consistency_clip_model, self.temporal_clip_consistency_preprocess = clip.load("ViT-B/32", device=self.device)
        if "temporal_motion_smoothness" in metrics:
            temporal_motion_smoothness_motion_model_config_path="third_party/amt/cfgs/AMT-S.yaml"
            temporal_motion_smoothness_motion_model_ckpt=os.path.join(ckpt_path,"amt_model/amt-s.pth")
            if not os.path.exists(temporal_motion_smoothness_motion_model_ckpt):
                wget_command = ['wget', '-P', os.path.dirname(temporal_motion_smoothness_motion_model_ckpt),
                                'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth']
                subprocess.run(wget_command, check=True)
            self.temporal_motion_smoothness_motion_model = MotionSmoothness(temporal_motion_smoothness_motion_model_config_path, temporal_motion_smoothness_motion_model_ckpt, self.device)
        if "dynamic_degree" in metrics:
            dynamic_degree_model_ckpt=os.path.join(ckpt_path,"raft_model/models/raft-things.pth")
            if not os.path.exists(dynamic_degree_model_ckpt):
                wget_command = ['wget', '-P', os.path.join(ckpt_path,"raft_model"), 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip']
                unzip_command = ['unzip', '-d', os.path.join(ckpt_path,"raft_model"), f'{ckpt_path}/raft_model/models.zip']
                remove_command = ['rm', '-r', os.path.join(ckpt_path,"raft_model/models.zip")]
                subprocess.run(wget_command, check=True)
                subprocess.run(unzip_command, check=True)
                subprocess.run(remove_command, check=True)
            self.dynamic_degree_frame_interval=1
            self.dynamic_degree_model=DynamicDegree(edict({"model":dynamic_degree_model_ckpt, "small":False, "mixed_precision":False, "alternate_corr":False}),device=self.device)
        if "tracking_strength" in metrics:
            self.tracking_strength_model_cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(self.device)
            self.tracking_strength_grid_size=10
            self.tracking_strength_frame_interval=1
        if set(['3D_consistency_num_pts','3D_consistency_num_inliers_F','3D_consistency_keep_ratio','3D_consistency_mean_err','3D_consistency_rmse'])&set(metrics):
            self.consistency_3D_interval_list=[60, 50, 40, 30, 20, 10] # default setting
            # self.consistency_3D_interval_list=[20, 10] # default setting
            self.consistency_3D_ransac_th=3
        if "aesthetic_quality" in metrics:
            self.aesthetic_quality_model=get_aesthetic_model(ckpt_path).to(self.device)
            self.aesthetic_quality_clip_model, self.aesthetic_quality_preprocess = clip.load('ViT-L/14', device=self.device)
        if "imaging_quality" in metrics:
            imaging_quality_model_ckpt=f'{ckpt_path}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
            if not os.path.isfile(imaging_quality_model_ckpt):
                wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(imaging_quality_model_ckpt)]
                subprocess.run(wget_command, check=True)
            self.imaging_quality_model=MUSIQ(pretrained_model_path=imaging_quality_model_ckpt)
            self.imaging_quality_model.to(self.device)
            self.imaging_quality_model.training = False
        if set(['camera_alignment','main_object_alignment','background_alignment','style_alignment','overall_consistency'])&set(metrics):
            text_video_consistency_model_viclip_ckpt=f'{ckpt_path}/ViCLIP/ViClip-InternVid-10M-FLT.pth'
            if not os.path.exists(text_video_consistency_model_viclip_ckpt):
                wget_command = ['wget', 'https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViClip-InternVid-10M-FLT.pth', '-P', os.path.dirname(text_video_consistency_model_viclip_ckpt)]
                subprocess.run(wget_command, check=True)
            text_video_consistency_model_viclip_tokenizerp_ckpt = os.path.join(ckpt_path, "ViCLIP/bpe_simple_vocab_16e6.txt.gz")
            if not os.path.exists(text_video_consistency_model_viclip_tokenizerp_ckpt):
                wget_command = ['wget', 'https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz', '-P', os.path.dirname(text_video_consistency_model_viclip_tokenizerp_ckpt)]
                subprocess.run(wget_command)

            self.text_video_consistency_model_viclip_tokenizer=SimpleTokenizer(text_video_consistency_model_viclip_tokenizerp_ckpt)
            self.text_video_consistency_model_viclip = ViCLIP(tokenizer= self.text_video_consistency_model_viclip_tokenizer, pretrain=text_video_consistency_model_viclip_ckpt).to(self.device)


    # temporal consistency
    def calculate_temporal_dino_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateTemporalDinoConsistency(self.temporal_dino_consistency_dino_model,store_image_folder,self.device)

    def calculate_temporal_clip_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateTemporalClipConsistency(self.temporal_clip_consistency_clip_model,self.temporal_clip_consistency_preprocess,store_image_folder,self.device)

    def calculate_temporal_motion_smoothness(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateMotionSmoothness(self.temporal_motion_smoothness_motion_model,store_image_folder,self.device)

    # temporal motion strength
    def calculate_dynamic_degree(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateDynamicDegree(self.dynamic_degree_model,store_image_folder,self.dynamic_degree_frame_interval)

    def calculate_tracking_strength(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        imgs=[os.path.join(store_image_folder, "frames_" + str(f) + ".png") for f in range(1,1+len(os.listdir(store_image_folder)))]
        frames = [iio.imread(img)[np.newaxis,...] for img in imgs[::self.tracking_strength_frame_interval]]  # plugin="pyav"
        video = torch.tensor(np.concatenate(frames)).permute(0, 3, 1, 2)[None].float().to(self.device) # B T C H W
        self.tracking_strength_model_cotracker(video_chunk=video, is_first_step=True, grid_size=self.tracking_strength_grid_size)  
        all_pred_tracks=[]
        for ind in range(0, video.shape[1] - self.tracking_strength_model_cotracker.step, self.tracking_strength_model_cotracker.step):
            pred_tracks, pred_visibility = self.tracking_strength_model_cotracker(
                video_chunk=video[:, ind : ind + self.tracking_strength_model_cotracker.step * 2]
            )  # B 
            all_pred_tracks.append(np.array(pred_tracks[0].cpu()))

        all_pred_tracks=np.concatenate(all_pred_tracks,0)
        all_pred_tracks=all_pred_tracks-all_pred_tracks[0]
        all_pred_tracks=np.linalg.norm(all_pred_tracks, axis=-1)
        return np.mean(all_pred_tracks)

    # 3D consistency
    def calculate_3D_consistency_num_pts(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        begin_frame = 0 
        end_frame = len(os.listdir(store_image_folder))

        err_list_pd=pd.DataFrame(columns=[
                        "3D_consistency_num_pts",
                        "3D_consistency_num_inliers_F",
                        "3D_consistency_keep_ratio",
                        "3D_consistency_mean_err",
                        "3D_consistency_rmse",
                    ])
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - begin_frame - interval_num) / 5)
            if match_inter <=0:
                print(f"can not match at interval={interval_num}")
                continue
            for i in range(0, end_frame, match_inter):
                if(i + interval_num > end_frame - 1): 
                    break
                left_id = i
                right_id = i + interval_num
                left_img_path  = os.path.join(store_image_folder, "frames_" + str(left_id+1) + ".png")
                right_img_path = os.path.join(store_image_folder, "frames_" + str(right_id+1) + ".png")
                mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, self.consistency_3D_ransac_th)
                # print(f"Correct point count after removing misalignments with the fundamental matrix RANSAC: {num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")
                err_list_pd.loc[len(err_list_pd.index)] = [num_pts,num_inliers_F,keep_rate,mean_error,rmse]
            
        err_list_pd=err_list_pd.mean()
        return err_list_pd["3D_consistency_num_pts"]

    def calculate_3D_consistency_num_inliers_F(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        begin_frame = 0 
        end_frame = len(os.listdir(store_image_folder))

        err_list_pd=pd.DataFrame(columns=[
                        "3D_consistency_num_pts",
                        "3D_consistency_num_inliers_F",
                        "3D_consistency_keep_ratio",
                        "3D_consistency_mean_err",
                        "3D_consistency_rmse",
                    ])
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - begin_frame - interval_num) / 5)
            if match_inter <=0:
                print(f"can not match at interval={interval_num}")
                continue
            for i in range(0, end_frame, match_inter):
                if(i + interval_num > end_frame - 1): 
                    break
                left_id = i
                right_id = i + interval_num
                left_img_path  = os.path.join(store_image_folder, "frames_" + str(left_id+1) + ".png")
                right_img_path = os.path.join(store_image_folder, "frames_" + str(right_id+1) + ".png")
                mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, self.consistency_3D_ransac_th)
                # print(f"Correct point count after removing misalignments with the fundamental matrix RANSAC: {num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")
                err_list_pd.loc[len(err_list_pd.index)] = [num_pts,num_inliers_F,keep_rate,mean_error,rmse]
            
        err_list_pd=err_list_pd.mean()
        return err_list_pd["3D_consistency_num_inliers_F"]

    def calculate_3D_consistency_keep_ratio(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        begin_frame = 0 
        end_frame = len(os.listdir(store_image_folder))

        err_list_pd=pd.DataFrame(columns=[
                        "3D_consistency_num_pts",
                        "3D_consistency_num_inliers_F",
                        "3D_consistency_keep_ratio",
                        "3D_consistency_mean_err",
                        "3D_consistency_rmse",
                    ])
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - begin_frame - interval_num) / 5)
            if match_inter <=0:
                print(f"can not match at interval={interval_num}")
                continue
            for i in range(0, end_frame, match_inter):
                if(i + interval_num > end_frame - 1): 
                    break
                left_id = i
                right_id = i + interval_num
                left_img_path  = os.path.join(store_image_folder, "frames_" + str(left_id+1) + ".png")
                right_img_path = os.path.join(store_image_folder, "frames_" + str(right_id+1) + ".png")
                mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, self.consistency_3D_ransac_th)
                # print(f"Correct point count after removing misalignments with the fundamental matrix RANSAC: {num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")
                err_list_pd.loc[len(err_list_pd.index)] = [num_pts,num_inliers_F,keep_rate,mean_error,rmse]
            
        err_list_pd=err_list_pd.mean()
        return err_list_pd["3D_consistency_keep_ratio"]

    def calculate_3D_consistency_mean_err(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        begin_frame = 0 
        end_frame = len(os.listdir(store_image_folder))

        err_list_pd=pd.DataFrame(columns=[
                        "3D_consistency_num_pts",
                        "3D_consistency_num_inliers_F",
                        "3D_consistency_keep_ratio",
                        "3D_consistency_mean_err",
                        "3D_consistency_rmse",
                    ])
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - begin_frame - interval_num) / 5)
            if match_inter <=0:
                print(f"can not match at interval={interval_num}")
                continue
            for i in range(0, end_frame, match_inter):
                if(i + interval_num > end_frame - 1): 
                    break
                left_id = i
                right_id = i + interval_num
                left_img_path  = os.path.join(store_image_folder, "frames_" + str(left_id+1) + ".png")
                right_img_path = os.path.join(store_image_folder, "frames_" + str(right_id+1) + ".png")
                mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, self.consistency_3D_ransac_th)
                # print(f"Correct point count after removing misalignments with the fundamental matrix RANSAC: {num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")
                err_list_pd.loc[len(err_list_pd.index)] = [num_pts,num_inliers_F,keep_rate,mean_error,rmse]
            
        err_list_pd=err_list_pd.mean()
        return err_list_pd["3D_consistency_mean_err"]

    def calculate_3D_consistency_rmse(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        begin_frame = 0 
        end_frame = len(os.listdir(store_image_folder))

        err_list_pd=pd.DataFrame(columns=[
                        "3D_consistency_num_pts",
                        "3D_consistency_num_inliers_F",
                        "3D_consistency_keep_ratio",
                        "3D_consistency_mean_err",
                        "3D_consistency_rmse",
                    ])
        for interval_num in self.consistency_3D_interval_list:
            match_inter = int((end_frame - begin_frame - interval_num) / 5)
            if match_inter <=0:
                print(f"can not match at interval={interval_num}")
                continue
            for i in range(0, end_frame, match_inter):
                if(i + interval_num > end_frame - 1): 
                    break
                left_id = i
                right_id = i + interval_num
                left_img_path  = os.path.join(store_image_folder, "frames_" + str(left_id+1) + ".png")
                right_img_path = os.path.join(store_image_folder, "frames_" + str(right_id+1) + ".png")
                mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, self.consistency_3D_ransac_th)
                # print(f"Correct point count after removing misalignments with the fundamental matrix RANSAC: {num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")
                err_list_pd.loc[len(err_list_pd.index)] = [num_pts,num_inliers_F,keep_rate,mean_error,rmse]
            
        err_list_pd=err_list_pd.mean()
        return err_list_pd["3D_consistency_rmse"]

    # video frame quality
    def calculate_aesthetic_quality(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateLaionAesthetic(self.aesthetic_quality_model,self.aesthetic_quality_clip_model,self.aesthetic_quality_preprocess,store_image_folder,self.device)

    def calculate_imaging_quality(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        return EvaluateImagingQuality(self.imaging_quality_model,store_image_folder,self.device)

    # text-video alignment
    def calculate_camera_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if camera_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, camera_caption)
        else:
            return None

    def calculate_main_object_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if main_object_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, main_object_caption)
        else:
            return None

    def calculate_background_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if background_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, background_caption)
        else:
            return None
    
    def calculate_style_alignment(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if style_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, style_caption)
        else:
            return None

    def calculate_overall_consistency(self,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        if dense_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, dense_caption)
        elif short_caption is not None:
            return EvaluateTextVideoConsistency(self.text_video_consistency_model_viclip, video_path, self.text_video_consistency_model_viclip_tokenizer, self.device, short_caption)
        else:
            return None


    def __call__(self,metric,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption):
        
        return eval(f"self.calculate_{metric}(store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption)")
        
