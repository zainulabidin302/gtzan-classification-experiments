from pydub import AudioSegment as ass
import os
ass.converter = "C:\\Users\\Zaighee\\Desktop\\ffmpeg-20171208-4678339-win64-static\\ffmpeg-20171208-4678339-win64-static\\bin\\ffmpeg.exe"
def main():
	export_dir = 'database'
	import_dir = 'genres'

	classes = os.listdir(import_dir)

	for clas in classes:
		clas_dir = os.path.join(import_dir, clas)
		songs = os.listdir(clas_dir)
		for song in songs:
			filename = os.path.join(clas_dir,song)
			res = ass.from_file(filename, format='au')
			if not os.path.isdir(os.path.join(export_dir, clas)):
				os.makedirs(os.path.join(export_dir, clas))
			res.export(os.path.join(export_dir, clas, song + '.wav'), format='wav')

main()