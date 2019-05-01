import wikipedia
import codecs

path = "/home/soyeon1771/SNFtools/Div400/devset/devsetkeywords/"
topics = read_topic(path)
topics[2] = 'Santa Maria della Salute'
topics[3] = 'Camposanto Monumentale'
topics[5] = 'Casas_Grandes'
topics[10] = 'Kaiser-Wilhelm-Denkmal an der Porta Westfalica'
topics[13] = 'Louvre Pyramid'
topics[17] = 'Quinta Vergara'

for topic in topics:
	with codecs.open("wiki.txt", 'a', encoding='utf8') as f:
		if topic == 'Kaiser-Wilhelm-Denkmal an der Porta Westfalica': 
			wikipedia.set_lang('de')
		else:
			wikipedia.set_lang('en')
		w = wikipedia.page(topic)
		print w.url
		f.write(w.content)



