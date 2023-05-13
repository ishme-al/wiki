import requests
import json
from queue import PriorityQueue

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

def get_text(article):
	r = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&format=json&titles={article}&prop=extracts&explaintext)')
	for page in json.loads(r.text)['query']['pages'].values():
		if 'extract' in page:
			return page['extract']

from doc import *

print('Welcome to wiki bot')
while True:
	print()
	print('Choose an option:')
	print('1. Query page')
	print('2. Find path')
	print('3. Similarity Score')
	print('4. Find by similarity')
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

			d1 = get_text(first)
			d2 = get_text(second)

			print(similarity(d1, d2))

		case 4:
			query = input('Query: ')
			print()
			print('Starting at wikipedia')

			q = PriorityQueue()
			seen = set()
			q.put((0, 'Wikipedia'))
			seen.add('Wikipedia')
			path = []
			last = -1

			while True:
				sim, s = q.get()
				sim = -sim

				if sim <= last:
					break

				last = sim
				path.append(s)
				print(f'{s}: {sim}')

				for t in outgoing(s)[:16]:
					if t in seen:
						continue
					seen.add(t)
					# print(t)
					text = get_text(t)
					if text:
						q.put((-similarity(query, text), t))
				
			print('Path:')
			for p in path:
				print(p, end=' -> ')
			print('\b\b\b\b    ')

		case 5:
			exit()
		
		case _:
			print('Invalid choice')