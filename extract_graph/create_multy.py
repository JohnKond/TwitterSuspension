import random

period = 'feb_mar'

quote = 'data/' + period +'/graph_quote_'+ period +'.tsv'
mention = 'data/' + period +'/graph_mention_'+ period +'.tsv'
retweet = 'data/' + period +'/graph_retweet_'+ period +'.tsv'
multy = 'graph_multy_' + period + '.tsv'

quote_lines = (open(quote,'r').readlines())
mention_lines = (open(mention,'r').readlines())
retweet_lines = (open(retweet,'r').readlines())

lines = quote_lines + mention_lines + retweet_lines

random.shuffle(lines)
random.shuffle(lines)
random.shuffle(lines)


f = open(multy,'w+')
f.writelines(lines)
f.close()





