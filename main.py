from pydub import AudioSegment
import simpleaudio as sa
import sys
sys.path.append('/Users/zhangyuxuan/Desktop/ffmpeg')

# 加载音频文件
audio = AudioSegment.from_file("test.mp3")

# 转换为原始音频数据
play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)

# 等待音频播放完
play_obj.wait_done()


