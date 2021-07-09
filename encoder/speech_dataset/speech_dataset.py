"""speech_dataset dataset."""

import os

import tensorflow as tf

import tensorflow_datasets as tfds

# TODO(speech_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.87
"""

# TODO(speech_dataset): BibTeX citation
_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""


_URL = "http://www.openslr.org/12"
_DL_URL = "http://www.openslr.org/resources/12/"
_DL_URLS = {
    "dev_clean": _DL_URL + "dev-clean.tar.gz",
    "dev_other": _DL_URL + "dev-other.tar.gz",
    "test_clean": _DL_URL + "test-clean.tar.gz",
    "test_other": _DL_URL + "test-other.tar.gz",
    "train_clean100": _DL_URL + "train-clean-100.tar.gz",
    "train_clean360": _DL_URL + "train-clean-360.tar.gz",
    "train_other500": _DL_URL + "train-other-500.tar.gz",
}


class SpeechDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for speech_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.',}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(speech_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(builder=self,
                                 description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like audio, transcript, speaker_id, chapter_id, audio_id ...
            'audio': tfds.features.Audio(sample_rate=16000),
            'text': tfds.features.Text(),
            'speaker_id': tf.int64,
            'chapter_id': tf.int64,
            'id': tf.string,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('audio', 'text'),  # Set to `None` to disable
        homepage=_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(speech_dataset): Downloads the data and defines the splits
    extracted_path = dl_manager.download_and_extract(_DL_URLS["dev_clean"])

    # TODO(speech_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return [tfds.core.SplitGenerator(name= tfds.Split.TRAIN, gen_kwargs= {'path':extracted_path})]

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(speech_dataset): Yields (key, example) tuples from the dataset
    transcripts_glob = os.path.join(path, 'LibriSpeech', '*/*/*/*.txt')
    for transcript_file in tf.io.gfile.glob(transcripts_glob):
      path = os.path.dirname(transcript_file)
      with tf.io.gfile.GFile(os.path.join(path, transcript_file)) as f:
        for line in f:
          line = line.strip()
          key, transcript = line.split(" ", 1)
          audio_file = "%s.flac" % key
          speaker_id, chapter_id = [int(el) for el in key.split("-")[:2]]
          example = {
              'audio': os.path.join(path, audio_file),
              'text': transcript,
              'speaker_id': int(speaker_id),
              'chapter_id': int(chapter_id),
              'id': key
          }
          yield key, example