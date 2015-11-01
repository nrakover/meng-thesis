import urllib

URL_PREFIX = 'http://www.image-net.org/api/download/imagenet.attributes.synset?wnid='
PATH_PREFIX = 'attributes_MAT_files'

f = open("synsets-with-attributes.txt", "r")

synsets = f.readlines()
progress_ticker = 1
for synset in synsets:
	# get the real URL of the MAT file
	url_connection = urllib.urlopen(URL_PREFIX + synset[:-1])
	initial_response = url_connection.read()
	url_connection.close()
	start_i = initial_response.find('url=')
	end_i = initial_response.find('" />')
	if start_i == -1 or end_i == -1:
		continue
	resolved_url = initial_response[start_i+4 : end_i]

	# retrieve the file
	urllib.urlretrieve(resolved_url, filename="{0}/{1}.attrann.mat".format(PATH_PREFIX, synset[:-1]))

	# progress update
	print '{0} %'.format(progress_ticker/len(synsets))
	progress_ticker += 1

f.close()
