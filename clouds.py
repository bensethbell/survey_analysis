from wordcloud import WordCloud
import pandas as pd
import sys

'''
must have wordcloud installed (pip install wordlcloud)
run in terminal:
python clouds.py [survey response file] [output file path]
'''
class PilotlyWordCloud:
	def __init__(self):
		pass

	def build_word_cloud(self, text, filepath = 'test.png', file_type = 'png'):
		wc = WordCloud(relative_scaling=.5).generate(text)
		image = wc.to_image()
		image.save(filepath, format = file_type)
		# plt.figure()
		# plt.imshow(wc)
		# plt.axis("off")
		# plt.show()

def get_text_from_rawfile(filepath, content_col = 'Response', min_length = 0):
	'''
	From raw answer csv will convert to prep for word_cloud
	'''
	df = pd.read_csv(filepath)
	dfnona = df.dropna(axis = 0,  subset = [content_col])
	#arr = dfnona[content_col].values
	arr = filter(lambda x: len(x.split()) > min_length, dfnona[content_col].values)
	return ' '.join(arr)




if __name__ == '__main__':
	if len(sys.argv) > 2:
		fp = str(sys.argv[2])
	else:
		fp = 'wordcloudtest.png'
	text = get_text_from_rawfile(str(sys.argv[1]))
	pwc = PilotlyWordCloud()
	pwc.build_word_cloud(text, filepath = fp)


	pass