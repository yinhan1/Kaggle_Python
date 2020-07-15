# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Consumer Complaints Classification with LSTM
# 
# This script looks into the consumer complaints from Bank of America in California. It applies LSTM on complaint classification and figures out the products consumers are not satisfied with. 

# %%
import pandas as pd 

OUTPUT_FOLDER = os.getcwd() + '/output/'

data = pd.read_excel('consumer complaints BOA_CA.xlsx')
df = (pd.DataFrame(data, columns = ['product','consumer_complaint_narrative'])
        .dropna() 
        .set_axis(['product', 'complaint'], axis = 1, inplace = False)
        .reset_index(drop = True))

## regroup product labels
df['product'] = df['product'].replace({
    'Consumer Loan': 'Consumer loan',
    'Payday loan': 'Consumer loan',
    'Student loan': 'Consumer loan',
    'Prepaid card': 'Bank account or service'
})

## create new feature: complaint length
df['length'] = df['complaint'].apply(lambda x: len(x))
df.head()

# %% [markdown]
# ## Text Pre-processing
# 
# 1. lower case
# 2. remove punctuations
# 3. tokenize to index.... 
# 

# %%
import re
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords

porter_stemmer = PorterStemmer()

## function to clean raw text
def clean_text(message):

    ### remove stop words
    message_no_stopwrd = remove_stopwords(message)

    ### simple preprocess
    tokens = simple_preprocess(message_no_stopwrd, deacc = True)

    ### stemming tokens
    tokens_stemmed = [porter_stemmer.stem(token) for token in tokens]

    return tokens_stemmed

## clean and tokenize raw text
df_token = df.assign(complaint = df['complaint'].apply(lambda x: clean_text(x)))
df_token.head()

# %% [markdown]
# ## Exploring Data
# 
# ### (1) Inbalanced distribution of product labels 

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
ax = sns.countplot(y = "product", data = df_token, order = df_token['product'].value_counts().index)
ax.set(xlabel = 'Number of Complaints', ylabel = '')
plt.show()

# %% [markdown]
# ### (2) Complaint length might help on prediction

# %%
ax = sns.boxplot(x = 'length', y = 'product', data = df_token, 
                 order = df_token['product'].value_counts().index)
plt.show()

# %% [markdown]
# ## Word Cloud 

# %%
from wordcloud import WordCloud

## function to generate word cloud
def word_cloud_plotter(complaint_column):
  
    ### generate word cloud
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color = 'white', 
                    min_font_size = 10).generate(str(complaint_column)) 
    
    ### plot word cloud
    plt.figure(figsize = (5, 5), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show() 

# %% [markdown]
# ### (1) Complaints on Mortgage

# %%
word_cloud_plotter(df_token.query('product == "Mortgage"').complaint)

# %% [markdown]
# ### (2) Complaints on Bank Account or Service

# %%
word_cloud_plotter(df_token.query('product == "Bank account or service"').complaint)


# %%
from gensim.models import Word2Vec

## function to train models for work embeddings
def save_wEmbed(model_type, tokens):

    ### set up output folder
    word2vec_model_file = OUTPUT_FOLDER + 'word2vec_' + model_type + '.model'

    ### set up parameters
    size = 1000
    window = 3
    min_count = 1
    workers = 3
    sg = int(model_type == 'sGram')

    ### train model
    w2v_model = Word2Vec(tokens, 
                         min_count = min_count, 
                         size = size, 
                         workers = workers, 
                         window = window, 
                         sg = 1)
    ### save model 
    w2v_model.save(word2vec_model_file)


# %%
## extract tokens to fit 
### stemmed_tokens = pd.Series(df_token['complaint']).values

## save CBOW
### save_wEmbed('CBOW', stemmed_tokens)

## save skip-gram
### save_wEmbed('sGram', stemmed_tokens)

# %% [markdown]
# ## Vectorize Text
# %% [markdown]
# ## LSTM
