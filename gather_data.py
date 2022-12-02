from bs4 import BeautifulSoup
from imdb import Cinemagoer
import csv
from urllib import request


possible_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17']


def get_rating(film):
    if film.get('certificates') is not None:
        for j in film.get('certificates'):
            if 'United States:G' in j['full']:
                return 'G'
            elif 'United States:PG' in j['full']:
                return 'PG'
            elif 'United States:PG-13' in j['full']:
                return 'PG-13'
            elif 'United States:R' in j['full']:
                return 'R'
            elif 'United States:NC-17' in j['full']:
                return 'NC-17'
    print(film['title'])
    return None


def get_parental_content(film_id):
    id_types = ['advisory-nudity', 'advisory-violence', 'advisory-profanity', 'advisory-alcohol', 'advisory-frightening', 'advisory-spoiler-nudity', 'advisory-spoiler-violence', 'advisory-spoiler-profanity', 'advisory-spoiler-alcohol', 'advisory-spoiler-frightening']
    request_url = request.urlopen('https://www.imdb.com/title/tt' + film_id + '/parentalguide?ref_=tt_stry_pg')
    soup = BeautifulSoup(request_url.read())
    votes_string = soup.find_all("span", class_="ipl-vote-button__details")
    votes = list(map(lambda x: int(x.string.replace(',', '')), votes_string))
    for i in range(0, len(votes), 4):
        total = votes[i] + votes[i+1] + votes[i+2] + votes[i+3]
        if total > 0:
            votes[i] /= total
            votes[i+1] /= total
            votes[i+2] /= total
            votes[i+3] /= total
    return votes


def get_with_spoiler(movie, type_of_guide):
    normal = movie.get(type_of_guide)
    spoiler = movie.get(type_of_guide.split(' ')[0] + 'spoiler' + type_of_guide.split(' ')[1])
    if normal is None:
        normal = []
    if spoiler is None:
        spoiler = []
    combined = " ".join(normal + spoiler)
    return combined


def get_without_spoiler(movie, type_of_guide):
    normal = movie.get(type_of_guide)
    if normal is None:
        normal = []
    combined = " ".join(normal)
    return combined

ia = Cinemagoer()
top = ia.get_top250_movies()
pop = ia.get_popular100_movies()
bot = ia.get_bottom100_movies()
with open('additional_movies.txt', 'rb') as f:
    additional_movies = f.read().splitlines()
count = 0
with open('dataset.tsv', 'a', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(
        ['title','rating', 'nudity_none', 'nudity_mild', 'nudity_moderate', 'nudity_severe', 'violence_none', 'violence_mild',
         'violence_moderate', 'violence_severe', 'profanity_none', 'profanity_mild', 'profanity_moderate',
         'profanity_severe', 'alcohol_none', 'alcohol_mild', 'alcohol_moderate', 'alcohol_severe', 'frightning_none',
         'frightning_mild', 'frightning_moderate', 'frightning_severe', 'content'])

movie_set = set()

for i in top:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    movie_set.add(title)
    rating = i.get('mpaa')
    if rating is None:
        rating = get_rating(i)
        if rating is None:
            continue
    else:
        rating = rating.split()[1]
    if rating not in possible_ratings:
        continue
    nudity_content = get_with_spoiler(i, 'advisory nudity')
    violence_content = get_with_spoiler(i, 'advisory violence')
    alcohol_content = get_without_spoiler(i, 'advisory alcohol')
    frightening_content = get_with_spoiler(i, 'advisory frightening')
    profanity_content = get_without_spoiler(i, 'advisory profanity')
    votes = get_parental_content(i.movieID)
    with open('dataset.tsv', 'a', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(
            [title, rating, votes[0], votes[1], votes[2], votes[3], votes[4], votes[5], votes[6], votes[7], votes[8], votes[9],
             votes[10], votes[11], votes[12], votes[13], votes[14], votes[15], votes[16], votes[17], votes[18],
             votes[19], nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

count = 0
for i in pop:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    if title not in movie_set:
        rating = i.get('mpaa')
        if rating is None:
            rating = get_rating(i)
            if rating is None:
                continue
        else:
            rating = rating.split()[1]
        if rating not in possible_ratings:
            continue
        nudity_content = get_with_spoiler(i, 'advisory nudity')
        violence_content = get_with_spoiler(i, 'advisory violence')
        alcohol_content = get_without_spoiler(i, 'advisory alcohol')
        frightening_content = get_with_spoiler(i, 'advisory frightening')
        profanity_content = get_without_spoiler(i, 'advisory profanity')
        votes = get_parental_content(i.movieID)
        try:
            with open('dataset.tsv', 'a', encoding='utf-8') as f:
                w = csv.writer(f, delimiter='\t')
                w.writerow(
                    [title, rating, votes[0], votes[1], votes[2], votes[3], votes[4], votes[5], votes[6], votes[7], votes[8],
                     votes[9], votes[10], votes[11], votes[12], votes[13], votes[14], votes[15], votes[16], votes[17],
                     votes[18], votes[19], nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' +
                     frightening_content + ' ' + profanity_content])
        except:
            print(len(votes))
            continue

count = 0
for i in bot:
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    if title not in movie_set:
        rating = i.get('mpaa')
        if rating is None:
            rating = get_rating(i)
            if rating is None:
                continue
        else:
            rating = rating.split()[1]
        if rating not in possible_ratings:
            continue
        nudity_content = get_with_spoiler(i, 'advisory nudity')
        violence_content = get_with_spoiler(i, 'advisory violence')
        alcohol_content = get_without_spoiler(i, 'advisory alcohol')
        frightening_content = get_with_spoiler(i, 'advisory frightening')
        profanity_content = get_without_spoiler(i, 'advisory profanity')
        votes = get_parental_content(i.movieID)
        with open('dataset.tsv', 'a', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(
                [title, rating, votes[0], votes[1], votes[2], votes[3], votes[4], votes[5], votes[6], votes[7], votes[8],
                 votes[9],
                 votes[10], votes[11], votes[12], votes[13], votes[14], votes[15], votes[16], votes[17], votes[18],
                 votes[19],
                 nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

count = 0
for movie in additional_movies:
    i = ia.get_movie(movie[2:])
    print(count)
    count += 1
    ia.update(i, info=['parents guide'])
    title = i['title']
    if title not in movie_set:
        rating = i.get('mpaa')
        if rating is None:
            rating = get_rating(i)
            if rating is None:
                continue
        else:
            rating = rating.split()[1]
        if rating not in possible_ratings:
            continue
        nudity_content = get_with_spoiler(i, 'advisory nudity')
        violence_content = get_with_spoiler(i, 'advisory violence')
        alcohol_content = get_without_spoiler(i, 'advisory alcohol')
        frightening_content = get_with_spoiler(i, 'advisory frightening')
        profanity_content = get_without_spoiler(i, 'advisory profanity')
        votes = get_parental_content(i.movieID)
        with open('dataset.tsv', 'a', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(
                [title, rating, votes[0], votes[1], votes[2], votes[3], votes[4], votes[5], votes[6], votes[7], votes[8],
                 votes[9],
                 votes[10], votes[11], votes[12], votes[13], votes[14], votes[15], votes[16], votes[17], votes[18],
                 votes[19],
                 nudity_content + ' ' + violence_content + ' ' + alcohol_content + ' ' + frightening_content + ' ' + profanity_content])

