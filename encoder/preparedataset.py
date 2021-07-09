from encoder import speech_dataset
import tensorflow_datasets as tfds
import tensorflow_io as tfio

N_frames = 120

def read_audio_file(filename):
    audio = filename['audio']
    audio = tf.cast(audio, dtype=tf.float32) / 32768.0
    return audio

def noise_removal(tensor):
    position = tfio.audio.trim(tensor, axis=0, epsilon=0.1)
    processed = tensor[position[0]:position[1]]
    return processed

def fade_in_out(tensor):
    return tfio.audio.fade(tensor, fade_in=1000, fade_out=2000, mode="logarithmic")

def wav_to_mel_spectrogram(tensor):
    spectrogram = tfio.audio.spectrogram(tensor, nfft=512, window=512, stride=128)
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=40, fmin=0, fmax=512)
    return mel_spectrogram

def read_speaker_identity(filepath):
    parts = tf.strings.split(filepath, sep=os.path.sep)
    return parts[-3]

def random_partial(input):
  """
  Crops the frames into a partial utterance of n_frames

  :return: the partial utterance frames and a tuple indicating the start and end of the
  partial utterance in the complete utterance.
  """
  frames = input
  n_frames = tf.constant(N_frames, dtype=tf.int32)
  MAXVAL = tf.size(frames[:,0]) - n_frames
  if  MAXVAL <= 0:
    start = 0
  else:
    start = tf.random.uniform(shape=(), maxval=MAXVAL, dtype=tf.int32)
  end = start + n_frames

  return frames[start:end], (start, end)

def get_waveform_speakerlabel(filename):
    wave = read_audio_file(filename)
    wave = noise_removal(wave)
    wave = wav_to_mel_spectrogram(wave)
    wave, position = random_partial(wave)
    speaker_identity = filename['speaker_id']
    return {'partial_utterance': wave, 'position': position, 'speaker_id': speaker_identity}

def load_make_dataset():
    dataset = tfds.load("speech_dataset")
    dataset = dataset.map(get_waveform_speakerlabel, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=32, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset