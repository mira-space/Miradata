# -*- coding: utf-8 -*-
import tqdm
import os
import youtube_dl
import argparse
import pandas as pd
import os, argparse
import tqdm
import requests
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--meta_csv", type=str,default="data/data_list/miradata_v1_9k.csv")
parser.add_argument("--video_start_id", type=int,default=0)
parser.add_argument("--video_end_id", type=int,default=100)
parser.add_argument("--raw_video_save_dir", type=str,default="data/raw_video")
parser.add_argument("--clip_video_save_dir", type=str,default="data/")
args = parser.parse_args()


df = pd.read_csv(args.meta_csv, encoding='utf-8')
print(f"Successfully loaded the csv file")

for i, row in tqdm.tqdm(df.iterrows()):
    download_id=int(row["clip_id"])
    if args.video_start_id>download_id or args.video_end_id<download_id:
        continue
    raw_video_download_path=os.path.join(args.raw_video_save_dir,
                                         str(download_id//1000).zfill(9),
                                         str(download_id).zfill(12)+".mp4")
    
    if not os.path.exists(raw_video_download_path):
        if not os.path.exists(os.path.dirname(raw_video_download_path)):
            os.makedirs(os.path.dirname(raw_video_download_path))

        # download
        fail_times=0
        while True:
            try:
                if "youtube" in row["source"]:
                    video_id=row["video_id"]

                    ydl_opts = {
                        'format': '22', # mp4        1280x720   720p  550k , avc1.64001F, 24fps, mp4a.40.2@192k (44100Hz) (best)
                        'continue': True,
                        'outtmpl': raw_video_download_path,
                        'external-downloader':'aria2c',
                        'external-downloader-args': '-x 16 -k 1M',
                    }
                    
                    ydl = youtube_dl.YoutubeDL(ydl_opts)
                    ydl.download([row["video_url"]])
                    break
                else:
                    res = requests.get(row["video_url"], stream=True)
                    if os.path.exists(raw_video_download_path+".tmp"):
                        os.remove(raw_video_download_path+".tmp")
                    with open(raw_video_download_path+".tmp", 'wb') as f:
                        for chunk in res.iter_content(chunk_size=10240):
                            f.write(chunk)
                    os.rename(raw_video_download_path+".tmp", raw_video_download_path)
                    break

            except Exception as error:
                print(error)
                print(f"Can not download video with download id: {download_id}")
                print(f"Try another time")
                fail_times+=1
                if fail_times==3:
                    print(f"Skip video with download id: {download_id}. Reach max download times.")
                    break

    # cut
    try:
        clip_video_path=os.path.join(args.clip_video_save_dir,row['file_path'])
        if os.path.exists(clip_video_path):
            continue
        run_ss=eval(row["timestamp"])[0]
        run_t=str(datetime.strptime(eval(row["timestamp"])[1], "%H:%M:%S.%f")-datetime.strptime(eval(row["timestamp"])[0], "%H:%M:%S.%f"))
        run_command=f"ffmpeg -ss {run_ss} -t {run_t} -i {raw_video_download_path} -c copy -y {clip_video_path}"
        
        if not os.path.exists(os.path.dirname(clip_video_path)):
            os.makedirs(os.path.dirname(clip_video_path))

        os.system(run_command)

    except Exception as error:
        print(error)
        print(f"error in cutting clip")

print(f"Finish")
