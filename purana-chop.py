import os
import numpy as np
import time
import scipy.io.wavfile as wav
def mapper(x):
	length = np.shape(x)
	sqrt_length = int(np.sqrt(length))
	x = x[:sqrt_length**2]
	x = x.reshape(sqrt_length, sqrt_length)
	return x

def main():
	start_time = time.time()
	export_dir = 'database'

	classes = os.listdir(export_dir)
	tmp = np.empty((0, 2))
	number_of_songs = 0
	number_of_classes = 0
	start_time = time.time()

	for clas in classes:
		number_of_classes += 1
		clas_dir = os.path.join(export_dir, clas)
		songs = os.listdir(clas_dir)
		for song in songs:
			number_of_songs += 1
			filename = os.path.join(clas_dir,song)
			samplerates, sample = wav.read(filename)
			# print(np.shape(sample))
			# for clip in spectrogram.audio_cutter(filename, clipsize=20000):
			start = 0
			second = (101**2) * 3 # w x h x d = 101x 101x 3

			while start != ((101**2) * 3) * 21:
				tmp = np.vstack((tmp, np.array([(np.array(sample[start:second], dtype=np.uint8)), clas])))
				start = second
				second += (101**2) * 3

		print('Processing class {}'.format(clas))
	np.save('database.npz', np.array(tmp))	
	print('Time taken: {}'.format(time.time() - start_time))
	print('DONE!')

main()