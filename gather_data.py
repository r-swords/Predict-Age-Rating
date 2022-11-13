from imdb import Cinemagoer
import csv


def get_with_spoiler(movie, type_of_guide):
    normal = movie.get(type_of_guide)
    spoiler = movie.get(type_of_guide.split(' ')[0] + 'spoiler' + type_of_guide.split(' ')[1])
    if normal is None:
        normal = []
    if spoiler is None:
        spoiler = []
    count = len(normal) + len(spoiler)
    combined = " ".join(normal + spoiler)
    return count, combined


def get_without_spoiler(movie, type_of_guide):
    normal = movie.get(type_of_guide)
    if normal is None:
        normal = []
    count = len(normal)
    combined = " ".join(normal)
    return count, combined

ia = Cinemagoer()
top = ia.get_top250_movies()
pop = ia.get_popular100_movies()
bot = ia.get_bottom100_movies()
count = 0
with open('dataset.tsv', 'a', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(
        ['title', 'rating', 'nudity_count', 'violence_count', 'alcohol_count', 'frightening_count', 'profanity_count',
         'content'])
movie_set = set()
for i in top:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    movie_set.add(title)
    rating = i.get('mpaa')
    if rating is None:
        continue
    rating = rating.split()[1]
    nudity_count, nudity_content = get_with_spoiler(i, 'advisory nudity')
    violence_count, violence_content = get_with_spoiler(i, 'advisory violence')
    alcohol_count, alcohol_content = get_without_spoiler(i, 'advisory alcohol')
    frightening_count, frightening_content = get_with_spoiler(i, 'advisory frightening')
    profanity_count, profanity_content = get_without_spoiler(i, 'advisory profanity')
    with open('dataset.tsv', 'a', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([title, rating, nudity_count, violence_count, alcohol_count, frightening_count, profanity_count,
                    nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

count = 0
for i in pop:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    if title not in movie_set:
        rating = i.get('mpaa')
        if rating is None:
            continue
        rating = rating.split()[1]
        nudity_count, nudity_content = get_with_spoiler(i, 'advisory nudity')
        violence_count, violence_content = get_with_spoiler(i, 'advisory violence')
        alcohol_count, alcohol_content = get_without_spoiler(i, 'advisory alcohol')
        frightening_count, frightening_content = get_with_spoiler(i, 'advisory frightening')
        profanity_count, profanity_content = get_without_spoiler(i, 'advisory profanity')
        with open('dataset.tsv', 'a', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow([title, rating, nudity_count, violence_count, alcohol_count, frightening_count, profanity_count,
                        nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

count = 0
for i in bot:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    if title not in movie_set:
        rating = i.get('mpaa')
        if rating is None:
            continue
        rating = rating.split()[1]
        nudity_count, nudity_content = get_with_spoiler(i, 'advisory nudity')
        violence_count, violence_content = get_with_spoiler(i, 'advisory violence')
        alcohol_count, alcohol_content = get_without_spoiler(i, 'advisory alcohol')
        frightening_count, frightening_content = get_with_spoiler(i, 'advisory frightening')
        profanity_count, profanity_content = get_without_spoiler(i, 'advisory profanity')
        with open('dataset.tsv', 'a', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow([title, rating, nudity_count, violence_count, alcohol_count, frightening_count, profanity_count,
                        nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

