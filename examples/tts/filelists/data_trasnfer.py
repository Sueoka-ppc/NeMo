import sys,os
from pydub import AudioSegment
import glob
import csv

with open("metadata.csv",'r',encoding="utf8")as f:
    reader = csv.reader(f,delimiter='|')
    list_data = []
    for entry in reader:
        print(entry)
        #{"audio_filepath": "/path/to/audio1.wav", "text": "the transcription", "duration": 0.82}
        # 音声ファイルの読み込み
        sound = AudioSegment.from_file(entry[0], "wave")

        # 情報の取得
        time = sound.duration_seconds # 再生時間(秒)
        
        data = '{"audio_filepath": '"\" filelists/"+entry[0]+"\""',"text": '"\""+entry[1]+"\""', "duration": '+str(time)+'} \r\n'

        list_data.append(data)

print(list_data)

with open("train.json","w",encoding='utf8') as fl:
    #writer = csv.writer(fl,quotechar='"', quoting=csv.QUOTE_NONE)
    #writer.writerows(list_data)
    fl.writelines(list_data)