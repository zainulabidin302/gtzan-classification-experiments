# gtzan-classification-experiments
ML-17 - University project - MLP, Convutional 2d Network


# Dataset 
dataset can be downloaded from http://marsyasweb.appspot.com/download/data_sets/. This dataset was used for the well known paper in genre classification " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

Unfortunately the database was collected gradually and very early on in my research so I have no titles (and obviously no copyright permission etc). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. Nevetheless I have been providing it to researchers upon request mainly for comparison purposes etc. Please contact George Tzanetakis (gtzan@cs.uvic.ca) if you intend to publish experimental results using this dataset.

The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. 

# To convert .au files to .wav use au_to_wav_converter.py 
  change **export_dir** and and **import_dir** to correct directories in the main function
  
  
# to generate samples of different sizes, use purana-chop.py

  change SAMPLE_SIZE and CHANNELS to get desired npz database of samples in the same directory.
  
 
