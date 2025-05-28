import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def process(filename):
    df = pd.read_csv(filename,encoding='iso-8859-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = vectorizer.fit_transform(df['message'])

    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    bow_df['label'] = df['label'].values

    return bow_df
