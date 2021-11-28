import soundfile as sf
import nemo 
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder


#model = nemo.core.ModelPT.load_from_checkpoint(checkpoint_path="./examples/tts/nemo_experiments/Tacotron2/2021-11-24_18-47-24/checkpoints/Tacotron2--val_loss=7.8324-epoch=3.ckpt")
# Download and load the pretrained fastpitch model
#spec_generator = SpectrogramGenerator.from_pretrained(model_name="tts_en_fastpitch").cuda()
spec_generator = SpectrogramGenerator.restore_from(restore_path="./examples/tts/nemo_experiments/Tacotron2/checkpoints/Tacotron2.nemo").cuda()
# Download and load the pretrained hifigan model
vocoder = Vocoder.from_pretrained(model_name="tts_waveglow_88m").cuda()

# All spectrogram generators start by parsing raw strings to a tokenized version of the string
parsed = spec_generator.parse("konnichiha, watashiha ai chandesu.")
# They then take the tokenized string and produce a spectrogram
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
# Finally, a vocoder converts the spectrogram to audio
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save the audio to disk in a file called speech.wav
# Note vocoder return a batch of audio. In this example, we just take the first and only sample.
sf.write("speech.wav", audio.to('cpu').detach().numpy()[0], 22050)







