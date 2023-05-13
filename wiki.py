import requests
import json

def outgoing(article):
	url = f'https://en.wikipedia.org/w/api.php?action=query&titles={article}&prop=links&pllimit=max&format=json&plnamespace=0'
	r = requests.get(url)
	r = json.loads(r.text)
	links = []

	if '-1' in r['query']['pages']:
		return []

	# print(article)
	for page in r['query']['pages'].values():
		links += [links['title'] for links in page['links']]

	while 'continue' in r:
		r = requests.get(url + '&plcontinue=' + r['continue']['plcontinue'])
		r = json.loads(r.text)
		for page in r['query']['pages'].values():
			links += [links['title'] for links in page['links']]
	
	return links

def incoming(article):
	url = f'https://en.wikipedia.org/w/api.php?action=query&titles={article}&prop=linkshere&lhlimit=max&format=json&lhnamespace=0'
	r = requests.get(url)
	r = json.loads(r.text)
	links = []

	for page in r['query']['pages'].values():
		links += [links['title'] for links in page['linkshere']]

	while 'continue' in r:
		r = requests.get(url + '&lhcontinue=' + r['continue']['lhcontinue'])
		r = json.loads(r.text)
		for page in r['query']['pages'].values():
			links += [links['title'] for links in page['linkshere']]
			
	return links


print('Welcome to wiki bot')
while True:
	print()
	print('Choose an option:')
	print('1. Query page')
	print('2. Find path')
	print('3. Similarity Score')
	print('4. Find path by similarity')
	print('5. Quit')
	choice = input('> ')
	print()
	try:
		choice = int(choice)
	except ValueError:
		print('Invalid choice')
		continue

	match choice:
		case 1:
			print('Article:')
			article = input('> ')
			print()

			# url = f'https://en.wikipedia.org/w/api.php?action=query&titles={article}&prop=links&pllimit=max&format=json'
			# r = requests.get(url)
			# r = json.loads(r.text)

			# if '-1' in r['query']['pages']:
			# 	print('Article not found')
			# 	continue

			# if 'normalized' in r['query']:
			# 	article = r['query']['normalized'][0]['to']

			print(article)
			
			print('Outgoing:', len(outgoing(article)))
			print('Incoming:', len(incoming(article)))


		case 2:
			source = input('From: ')
			target = input('To: ')
			parent = {source: None}
			next = [source]
			i = 1
			found = False

			while target not in parent:
				print(f'Iteration {i}: {len(parent)} links')
				i += 1

				temp = []
				for s in next:
					if found:
						break
					for t in outgoing(s):
						if t == target:
							found = True
							parent[t] = s
							break
						if t not in parent:
							parent[t] = s
							temp.append(t)
				next = temp
				
			
			path = [target]
			while path[-1] != source:
				path.append(parent[path[-1]])
			
			print('Path:')
			for p in reversed(path):
				print(p, end=' -> ')
			print('\b\b\b\b    ')

		case 3:
			first = input('First article: ')
			second = input('Second article: ')

		case 4:
			pass

		case 5:
			exit()
		
		case _:
			print('Invalid choice')