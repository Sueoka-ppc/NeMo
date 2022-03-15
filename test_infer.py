import soundfile as sf
import datetime
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.collections.tts.models import MelGanModel
from nemo.collections.tts.models import WaveGlowModel

# Download and load the pretrained fastpitch model
spec_generator = SpectrogramGenerator.restore_from(restore_path="./examples/tts/glow_tts_train_ptm01/glow_tts_train01/checkpoints/glow_tts_train01.nemo").cuda()
# Download and load the pretrained hifigan model
#vocoder = Vocoder.from_pretrained(model_name="tts_hifigan").cuda()

#vocoder = MelGanModel.from_pretrained(model_name="tts_melgan").cuda()
vocoder = WaveGlowModel.from_pretrained(model_name="tts_waveglow_268m").cuda()

# All spectrogram generators start by parsing raw strings to a tokenized version of the string
parsed = spec_generator.parse("koNnnichiwA.watashiwaeiaichandesu.")
# They then take the tokenized string and produce a spectrogram
spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
# Finally, a vocoder converts the spectrogram to audio
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save the audio to disk in a file called speech.wav
# Note vocoder return a batch of audio. In this example, we just take the first and only sample.
timeS = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
sf.write("speech"+timeS+".wav", audio.to('cpu').detach().numpy()[0], 22050)
